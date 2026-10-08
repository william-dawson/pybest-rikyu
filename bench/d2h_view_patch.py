"""Return a view instead of a copy, for the 96% of transfers where it is safe.

Our pinned-buffer patch pays an extra host-to-host copy on every transfer,
because the buffer is reused and some callers RETAIN the array they are given.
Job 161947 priced that copy: a staged copy runs at 10.8 GB/s where the same DMA
into memory the engine can address runs at 193 GB/s.

Job 162176 then showed the retaining callers barely matter. Attributing every
move_tensor_to_cpu by caller line:

    td_GPU_helper:1487   95.6% of bytes at 240 AO, 96.0% at 580 AO   `dest[...] +=`
    c_splitting:998       4.0% at 580 AO                             `dest[...] +=`
    c_splitting:925       4.4% at 240 AO, 0.0% at 580 AO             RETURNED
    c_splitting:1158      0.0% (61 calls, zero bytes)                RETURNED

The `+=` sites consume the array inside the very expression that receives it, so
a view into a reused buffer is safe there. Only the sites that return it need a
copy, and they carry 4.4% of the volume at worst.

So: look at the caller, return a view where that is safe, copy where it is not.
Anything unrecognised copies -- the default is the safe one, so a PyBEST whose
line numbers have moved gets correctness, not corruption.

This is a benchmark patch, not a proposal. Keying on line numbers is version-
specific by nature. The point it demonstrates is the upstream one: at
crosslib_batching.py:1487 PyBEST already holds the destination, so the DMA could
write it directly and the staging buffer would not exist. One line, 96% of the
traffic.

Set PYBEST_D2H_VIEW=1. Requires the pinned patch to be active.
"""
from __future__ import annotations

import os
import sys

# Caller lines that consume the array inside the receiving expression.
SAFE_LINES = frozenset({1487, 998, 1062, 489, 437, 541, 1117})
# Caller lines that return it; these must keep copying.
COPY_LINES = frozenset({351, 582, 925, 1158})


def install(verbose: bool = True) -> bool:
    try:
        import pybest.linalg._gpu_support as gs
    except ImportError as exc:                              # noqa: BLE001
        if verbose:
            print(f"# d2h_view: not installed ({exc})", flush=True)
        return False

    backend = getattr(gs, "gpu_backend_select", None)
    if backend not in ("cupy", "pytorch"):
        if verbose:
            print(f"# d2h_view: no GPU backend selected ({backend})", flush=True)
        return False

    stats = {"view": 0, "copy": 0, "unknown": 0}

    if backend == "pytorch":
        import torch as _t

        pin: dict = {}

        def staged(x, copy: bool):
            n = x.numel()
            b = pin.get(x.dtype)
            if b is None or b.numel() < n:
                b = _t.empty(max(n, 2 * (0 if b is None else b.numel())),
                             dtype=x.dtype, pin_memory=True)
                pin[x.dtype] = b
            v = b[:n].view(x.shape)
            v.copy_(x, non_blocking=True)
            _t.cuda.synchronize()
            arr = v.numpy()
            return arr.copy() if copy else arr
    else:
        import cupy as _cp
        import numpy as _np

        orig = gs._ops["cupy"]["get_numpy"]
        cpin: dict = {}

        def staged(x, copy: bool):
            try:
                n = int(x.size)
                e = cpin.get(x.dtype)
                if e is None or e[0] < n:
                    sz = max(n, 2 * (e[0] if e else 0))
                    m = _cp.cuda.alloc_pinned_memory(sz * x.dtype.itemsize)
                    cpin[x.dtype] = (sz, m,
                                     _np.frombuffer(m, dtype=x.dtype, count=sz))
                    e = cpin[x.dtype]
                v = e[2][:n].reshape(x.shape)
                x.get(out=v)
                return v.copy() if copy else v
            except Exception as exc:                        # noqa: BLE001
                print(f"# d2h_view: cupy path failed ({exc})", flush=True)
                return orig(x)

    def dispatch(x):
        # Walk up to the first frame inside crosslib_batching rather than
        # assuming a depth: PyBEST reaches here through move_tensor_to_cpu or
        # through get_numpy_array, and a fixed depth would silently read the
        # wrong frame if that ever changes.
        line = None
        depth = 1
        while depth <= 6:
            try:
                f = sys._getframe(depth)
            except ValueError:
                break
            if f.f_code.co_filename.endswith("crosslib_batching.py"):
                line = f.f_lineno
                break
            depth += 1
        if line in SAFE_LINES:
            stats["view"] += 1
            return staged(x, copy=False)
        if line in COPY_LINES:
            stats["copy"] += 1
        else:
            stats["unknown"] += 1          # unrecognised: copy, the safe default
        return staged(x, copy=True)

    gs._ops[backend]["get_numpy"] = dispatch
    gs._ops[backend]["as_numpy"] = dispatch

    import atexit

    atexit.register(lambda: print(
        f"# d2h_view: returned a view {stats['view']} times, copied "
        f"{stats['copy']} (retaining caller) + {stats['unknown']} (unrecognised "
        f"caller, copied to be safe)", flush=True))
    if verbose:
        print("# d2h_view: view where the caller consumes it, copy where it "
              "escapes", flush=True)
    return True
