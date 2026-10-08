#!/usr/bin/env python3
"""Test host-memory pinning strategies without rebuilding the container.

Nsight Systems (job 161307, 240 AO) found both backends spending far more time
on memory management than on arithmetic, in opposite ways:

  CuPy      cudaHostAlloc 12.74 s + cudaFreeHost 2.60 s over 1032 calls, against
            3.49 s of GPU kernels. It pins correctly but re-pins constantly.
  PyTorch   no cudaHostAlloc at all, so transfers target pageable memory where
            cudaMemcpyAsync is synchronous: D2H 21.46 s against CuPy's 0.52 s
            for the same 65.8 GB.

Each has a different one-line-ish cause, patched here by monkey-patching the
backend op table rather than modifying the read-only image.

  cupy     `_cupy_clean` does
               get_default_memory_pool().free_all_blocks()
               get_default_pinned_memory_pool().free_all_blocks()
           and `clean_memory()` is called dozens of times per contraction, so
           every call discards the pinned pool that exists precisely to avoid
           re-pinning. The variant keeps the device-pool flush and drops the
           pinned one.

  pytorch  `as_numpy`/`get_numpy` is `lambda t: t.cpu().numpy()`, whose
           destination is ordinary pageable numpy memory, where
           cudaMemcpyAsync is synchronous. The variant copies into a REUSED
           pinned staging buffer first.

Both patches reach the synthetic ladder as well as molecular CCSD: c_splitting
calls `clean_memory` at 37 sites, and returns results via `move_tensor_to_cpu`,
which is `_get_ops()["get_numpy"]`. Patching the op table therefore covers both.

Measured at 240 AO (job 161361), PyTorch baseline vs pinned:
  CC total 97.62 -> 61.29 s (-37%), GPU: Generic 60.0 -> 24.7 s (-59%),
  same energy. The CuPy comparison there was confounded by its baseline cell
  taking the dump_cache path.

  python pin_experiment.py --driver systems/h2o10_cc-pvdz_ccsd.py \
                           --strategy pinned
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys


def patch_cupy_pinned_staging() -> str:
    """Return results through a single REUSED pinned host buffer.

    This is the BOUNDED fix, and the distinction matters. Simply making
    `clean_memory` stop flushing CuPy's pinned pool also removes the re-pinning,
    but a pool grows without limit: measured at N=800 it gained 3.4% and cost
    43 GiB of host memory, which is a poor trade on a memory-constrained code.
    One explicitly sized buffer gives the same benefit at no memory cost, which
    is how the PyTorch fix behaves (-17% to -37%, peak unchanged).

    `_ops["cupy"]["get_numpy"]` is `lambda t: t.get()`, which allocates a fresh
    pageable array and stages through the pool. Here we keep one pinned buffer
    per dtype, grown geometrically, and copy out of it.
    """
    import cupy as cp
    import numpy as np
    import pybest.linalg._gpu_support as gs

    orig = gs._ops["cupy"]["get_numpy"]
    cache: dict = {}

    def via_pinned(t):
        try:
            n = int(t.size)
            ent = cache.get(t.dtype)
            if ent is None or ent[0] < n:
                size = max(n, 2 * (ent[0] if ent else 0))
                mem = cp.cuda.alloc_pinned_memory(size * t.dtype.itemsize)
                arr = np.frombuffer(mem, dtype=t.dtype, count=size)
                cache[t.dtype] = (size, mem, arr)
                ent = cache[t.dtype]
            view = ent[2][:n].reshape(t.shape)
            t.get(out=view)
            # Copy out: the caller must not alias a buffer we will overwrite.
            return view.copy()
        except Exception as exc:                                  # noqa: BLE001
            # Never let the patch break a measurement; fall back and say so.
            print(f"# pin: cupy pinned path failed ({exc}), using t.get()",
                  flush=True)
            return orig(t)

    gs._ops["cupy"]["get_numpy"] = via_pinned
    gs._ops["cupy"]["as_numpy"] = via_pinned
    return "cupy get_numpy/as_numpy copy via a reused pinned staging buffer"


def patch_pytorch_pinned_staging() -> str:
    import torch
    import pybest.linalg._gpu_support as gs

    cache: dict = {}

    def as_numpy_via_pinned(t):
        n = t.numel()
        buf = cache.get(t.dtype)
        if buf is None or buf.numel() < n:
            # Grow geometrically so this settles after a few reallocations.
            buf = torch.empty(max(n, 2 * (buf.numel() if buf is not None else 0)),
                              dtype=t.dtype, pin_memory=True)
            cache[t.dtype] = buf
        view = buf[:n].view(t.shape)
        view.copy_(t, non_blocking=True)
        torch.cuda.synchronize()
        # Copy out: the caller must not alias a buffer we will overwrite. A
        # host-to-host copy of a few hundred MB costs single-digit ms, against
        # the 620 ms blocking transfers this replaces.
        return view.numpy().copy()

    gs._ops["pytorch"]["as_numpy"] = as_numpy_via_pinned
    gs._ops["pytorch"]["get_numpy"] = as_numpy_via_pinned
    return "pytorch: as_numpy/get_numpy copy via a reused pinned staging buffer"


def patch_cached_vram() -> str:
    """Query free VRAM once per process instead of on every call.

    `c_splitting` sizes its batches from `memory_usage()`, which is driver-level
    free VRAM and so FALLS as the allocator fills: 183.3 GiB on the first pass
    and 138.1 GiB on the second at N=1200. The second pass then splits harder
    and runs 5.7% slower (CuPy) or 18.0% slower (PyTorch) on identical input,
    and in a converged calculation every iteration after the first pays it.

    The same reading also decides the `ecfd < 0.4 * memhave` branch, whose
    crossing sits at N~1082 -- so a transient allocator state moves a hard
    performance cliff. Caching the first (cold-card) value removes both effects
    and costs nothing.
    """
    import pybest.linalg.crosslib_batching as _cb

    orig = _cb.memory_usage
    cache: dict = {}

    def cached():
        if "v" not in cache:
            cache["v"] = orig()
            print(f"# cachedvram: pinned at {cache['v'] / 2**30:.2f} GiB",
                  flush=True)
        return cache["v"]

    _cb.memory_usage = cached
    return "memory_usage() cached for the process lifetime"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True, help="generated driver to run")
    ap.add_argument("--strategy",
                    choices=("baseline", "pinned", "cachedvram", "both"),
                    required=True)
    # Everything after `--` goes to the driver, so this can also run
    # repro_table4.py, which parses its own --nbasis/--nocc/--reps.
    args, passthrough = ap.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    # Import the op table first so the patch lands before any contraction runs.
    import pybest.linalg._gpu_support as gs

    backend = gs.gpu_backend_select
    print(f"# pin_experiment backend={backend} strategy={args.strategy}", flush=True)
    if backend is None:
        print("# WARNING no GPU backend active; patch is a no-op", flush=True)

    if args.strategy in ("pinned", "both"):
        if backend == "cupy":
            print(f"# patch: {patch_cupy_pinned_staging()}", flush=True)
        elif backend == "pytorch":
            print(f"# patch: {patch_pytorch_pinned_staging()}", flush=True)
    if args.strategy in ("cachedvram", "both"):
        print(f"# patch: {patch_cached_vram()}", flush=True)

    sys.argv = [args.driver, *passthrough]
    if passthrough:
        print(f"# driver args: {' '.join(passthrough)}", flush=True)
    runpy.run_path(args.driver, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
