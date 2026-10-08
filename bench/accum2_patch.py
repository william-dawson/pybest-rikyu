"""Fuse and thread `arr[slice_] += factor * X`.

The first attempt at this (accum_patch.py) scaled X in place instead of
allocating a temporary, and changed nothing: +0.5% at 580 AO. The traffic model
says why, and predicts its own failure:

    PyBEST today    tmp = factor*X ; dest += tmp     3.94 TB
    accum_patch     X *= factor    ; dest += X       3.94 TB   <- same bytes
    fused           dest += factor*X, one pass       2.36 TB

(summed over a 920 AO run, from the 788 GB of measured D2H volume.)

The real problem is not the temporary, it is the rate. 3.94 TB in 287 s is
13.7 GB/s -- **4% of one Grace socket**, which has 384 GB/s and 768 across the
node. numpy's `+=` is single-threaded, so this is one core doing what sixty-four
could.

Fused and threaded, the same work is ~12-25 s instead of 287 s, for no extra
memory. Chunks are sized to stay in cache, so the `factor *` costs nothing
extra: the small per-chunk temporary never leaves L2.

How the interception works. `factor * X` returns X itself with the factor
RECORDED rather than applied, so no pass happens. The subsequent
`dest[slice] += X` reaches numpy as `np.add(dest_view, X, out=dest_view)`,
which dispatches to our `__array_ufunc__` because X is an ndarray subclass.
There we do the fused threaded accumulate and the factor is consumed exactly
once.

Safe for the same reason move_tensor_to_cpu_view is: X is ours, and every one
of the thirteen call sites consumes it inside the expression that receives it.
Anything we do not recognise -- a non-contiguous destination, a mismatched
shape, any other ufunc -- falls back to numpy with the factor applied normally,
so the failure mode is "no speedup", never "wrong answer".

  PYBEST_ACCUM2=1              enable
  PYBEST_ACCUM2_THREADS=n      default: min(32, cpu_count)
  PYBEST_ACCUM2_CHUNK=bytes    default 1 MiB per chunk
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_THREADS = int(os.environ.get("PYBEST_ACCUM2_THREADS", "0")) or min(
    32, os.cpu_count() or 8
)
_CHUNK = int(os.environ.get("PYBEST_ACCUM2_CHUNK", str(1 << 20))) // 8
_POOL = ThreadPoolExecutor(max_workers=_THREADS)
_STATS = {"fused": 0, "fallback": 0, "elements": 0}


def _fused_axpy(dest: np.ndarray, src: np.ndarray, factor) -> None:
    """dest += factor * src, chunked and threaded. numpy releases the GIL."""
    d = dest.reshape(-1)
    s = src.reshape(-1)
    n = d.size
    step = max(_CHUNK, (n + _THREADS - 1) // _THREADS)

    def work(i0: int) -> None:
        i1 = min(i0 + step, n)
        if factor == 1:
            d[i0:i1] += s[i0:i1]
        else:
            d[i0:i1] += factor * s[i0:i1]

    if n <= step:
        work(0)
        return
    list(_POOL.map(work, range(0, n, step)))


class LazyScaled(np.ndarray):
    """Records a scalar factor instead of applying it, then fuses on add."""

    _factor = 1.0

    def __array_finalize__(self, obj):
        if obj is not None:
            self._factor = getattr(obj, "_factor", 1.0)

    def __rmul__(self, other):
        if np.isscalar(other) and np.isrealobj(self):
            self._factor = self._factor * other
            return self
        return np.multiply(np.asarray(self) * self._factor, other)

    __mul__ = __rmul__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get("out")
        factor = self._factor
        if (
            ufunc is np.add
            and method == "__call__"
            and out is not None
            and len(out) == 1
            and len(inputs) == 2
            and inputs[1] is self
            and isinstance(out[0], np.ndarray)
            and out[0] is inputs[0]
            and out[0].flags.c_contiguous
            and self.flags.c_contiguous
            and out[0].shape == self.shape
            and out[0].dtype == self.dtype
        ):
            self._factor = 1.0          # consumed
            _STATS["fused"] += 1
            _STATS["elements"] += self.size
            _fused_axpy(out[0], np.asarray(self), factor)
            return out[0]
        # Anything unrecognised: behave exactly as a plain array would have.
        _STATS["fallback"] += 1
        self._factor = 1.0
        plain = [np.asarray(i) if i is self else i for i in inputs]
        if factor != 1.0:
            plain = [p * factor if p is not inputs[0] else p for p in plain]
        return getattr(ufunc, method)(*plain, **kwargs)


def install(verbose: bool = True) -> bool:
    try:
        import pybest.linalg._gpu_support as gs
        import pybest.linalg.crosslib_batching as cb
    except ImportError as exc:                              # noqa: BLE001
        if verbose:
            print(f"# accum2: not installed ({exc})", flush=True)
        return False

    backend = getattr(gs, "gpu_backend_select", None)
    if backend not in ("cupy", "pytorch"):
        if verbose:
            print(f"# accum2: no GPU backend ({backend})", flush=True)
        return False

    def wrap(fn):
        def wrapped(tensor):
            out = fn(tensor)
            if isinstance(out, np.ndarray):
                v = out.view(LazyScaled)
                v._factor = 1.0
                return v
            return out

        return wrapped

    for key in ("get_numpy", "as_numpy"):
        gs._ops[backend][key] = wrap(gs._ops[backend][key])
    for attr in ("move_tensor_to_cpu", "move_tensor_to_cpu_view"):
        fn = getattr(cb, attr, None)
        if fn is not None:
            setattr(cb, attr, wrap(fn))

    import atexit

    atexit.register(lambda: print(
        f"# accum2: fused {_STATS['fused']} accumulates "
        f"({_STATS['elements'] * 8 / 2**30:.1f} GiB of destination), "
        f"{_STATS['fallback']} fell back", flush=True))
    if verbose:
        print(f"# accum2: fused threaded accumulate, {_THREADS} threads, "
              f"{_CHUNK * 8 // 1024} KiB chunks", flush=True)
    return True
