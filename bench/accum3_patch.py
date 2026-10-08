"""Fused, threaded `arr[slice_] += factor * X` through PyTorch.

Two earlier attempts and what they taught:

  accum_patch    scaled X in place instead of allocating a temporary.
                 +0.5%: the traffic is identical either way, 3.94 TB.
  accum2_patch   chunked the accumulate over a thread pool. 12% faster and
                 NUMERICALLY WRONG -- the energy moved by 3.9e-4 Ha. The bug was
                 in the fallback branch, which scaled every operand that was not
                 literally inputs[0] rather than only the operand that was self,
                 so `np.add(self, other)` scaled `other` too. 150 of 424
                 accumulates took that path.

This version fixes that by construction and uses torch for the fast path:

  torch.from_numpy(dest).add_(torch.from_numpy(src), alpha=factor)

which is a real fused axpy -- 2.36 TB instead of 3.94 -- threaded by ATen with
its own grain sizing, and zero-copy in both directions because from_numpy
shares the buffer.

CORRECTNESS, since that is what went wrong last time:

  * The fast path fires only on the exact pattern np.add(dest, self, out=dest)
    with both operands contiguous, same shape and dtype, and dest the first
    input. Anything else falls back.
  * The fallback materialises SELF as a plain, already-scaled array and reruns
    the ufunc with every other operand untouched. There is no operand rewriting
    left to get wrong.
  * The pending factor is consumed exactly once: read and reset at the top of
    __array_ufunc__, before any branch.
  * __array_finalize__ does NOT propagate a pending factor to views. A slice of
    a scaled array would otherwise carry the factor and apply it twice.
  * PYBEST_ACCUM3_VERIFY=k checks the first k fused operations against a numpy
    reference and reports the largest deviation. Cheap insurance that costs
    nothing after the first few calls.

Only installs when PyTorch is the selected backend. PyBEST supports CuPy-only
installations, and a numpy accumulate that requires a second deep-learning
framework is not something to impose on them.

  PYBEST_ACCUM3=1
  PYBEST_ACCUM3_THREADS=n     default: leave ATen's own choice alone
  PYBEST_ACCUM3_VERIFY=k      default 0
"""
from __future__ import annotations

import os

import numpy as np

_VERIFY = int(os.environ.get("PYBEST_ACCUM3_VERIFY", "0"))
# 16 beat 64 at 580 AO (494.80 against 508.43 s): ATen's pool oversubscribes
# against the threads driving the GPU. Override with PYBEST_ACCUM3_THREADS.
_DEFAULT_THREADS = 16
_STATS = {"fused": 0, "fallback": 0, "bytes": 0, "maxdev": 0.0, "checked": 0}


def _torch_axpy(dest: np.ndarray, src: np.ndarray, factor) -> None:
    """dest += factor * src, fused and threaded, sharing both buffers."""
    import torch

    torch.from_numpy(dest).add_(torch.from_numpy(src), alpha=factor)


class LazyScaled(np.ndarray):
    """Defers `factor * X` so the following `+=` can fuse it."""

    _factor = 1.0

    def __array_finalize__(self, obj):
        # Deliberately NOT inherited: a view of a pending-scaled array would
        # otherwise carry the factor and apply it a second time.
        self._factor = 1.0

    def __rmul__(self, other):
        if np.isscalar(other) and np.isrealobj(self):
            self._factor = self._factor * other
            return self
        return np.multiply(np.asarray(self) * self._factor, other)

    __mul__ = __rmul__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        factor = self._factor
        self._factor = 1.0                      # consumed exactly once
        out = kwargs.get("out")
        if (
            ufunc is np.add
            and method == "__call__"
            and len(inputs) == 2
            and inputs[1] is self
            and out is not None
            and len(out) == 1
            and out[0] is inputs[0]
            and isinstance(out[0], np.ndarray)
            and out[0].flags.writeable
            and out[0].shape == self.shape
            and out[0].dtype == self.dtype == np.float64
            # torch.from_numpy needs positive strides; it is happy with
            # non-contiguous arrays otherwise, and base.py's destinations are
            # slices of a four-index array that are never C-contiguous. The
            # first version demanded contiguity and so fused only
            # td_GPU_helper, missing all twelve base.py sites (150 of 424
            # operations fell back).
            and all(st > 0 for st in out[0].strides)
            and all(st > 0 for st in self.strides)
        ):
            dest = out[0]
            src = np.asarray(self)
            if _VERIFY and _STATS["checked"] < _VERIFY:
                want = dest + factor * src
                _torch_axpy(dest, src, factor)
                dev = float(np.max(np.abs(dest - want))) if dest.size else 0.0
                _STATS["maxdev"] = max(_STATS["maxdev"], dev)
                _STATS["checked"] += 1
            else:
                _torch_axpy(dest, src, factor)
            _STATS["fused"] += 1
            _STATS["bytes"] += dest.nbytes
            return dest

        # Fallback: scale ONLY the operand that is self, change nothing else.
        _STATS["fallback"] += 1
        base = np.asarray(self)
        scaled = base if factor == 1 else base * factor
        new_inputs = tuple(scaled if i is self else i for i in inputs)
        return getattr(ufunc, method)(*new_inputs, **kwargs)


def install(verbose: bool = True) -> bool:
    try:
        import pybest.linalg._gpu_support as gs
        import pybest.linalg.crosslib_batching as cb
    except ImportError as exc:                              # noqa: BLE001
        if verbose:
            print(f"# accum3: not installed ({exc})", flush=True)
        return False

    if getattr(gs, "gpu_backend_select", None) != "pytorch":
        if verbose:
            print("# accum3: PyTorch is not the selected backend, skipping "
                  "(CuPy-only installs must not need torch)", flush=True)
        return False

    import torch

    nt = int(os.environ.get("PYBEST_ACCUM3_THREADS", "0")) or _DEFAULT_THREADS
    torch.set_num_threads(nt)

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
        gs._ops["pytorch"][key] = wrap(gs._ops["pytorch"][key])
    for attr in ("move_tensor_to_cpu", "move_tensor_to_cpu_view"):
        fn = getattr(cb, attr, None)
        if fn is not None:
            setattr(cb, attr, wrap(fn))

    import atexit

    def report() -> None:
        msg = (f"# accum3: fused {_STATS['fused']} "
               f"({_STATS['bytes'] / 2**30:.1f} GiB), "
               f"{_STATS['fallback']} fell back")
        if _STATS["checked"]:
            msg += (f", verified {_STATS['checked']} against numpy, "
                    f"max deviation {_STATS['maxdev']:.3e}")
        print(msg, flush=True)

    atexit.register(report)
    if verbose:
        print(f"# accum3: torch fused axpy, {torch.get_num_threads()} threads"
              + (f", verifying first {_VERIFY}" if _VERIFY else ""), flush=True)
    return True
