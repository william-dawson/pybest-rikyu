"""Remove the temporary in `arr[slice_] += factor * X`.

cProfile at 920 AO (job 162831) puts 287.1 s -- 16% of a 1827 s CCSD -- in the
OWN time of two functions:

    base.py:545(contract)                142.2 s over 473 calls
    crosslib_batching.py:1320(td_GPU_helper)  144.9 s over 243 calls

That is not interpreter overhead at 300-600 ms per call. It is numpy bulk
arithmetic attributed to the enclosing frame, because `+=` on a slice is
bytecode rather than a profiled call. The idiom appears at twelve sites in
base.py (689, 696, 699, 702, 708, 728, 730, 732, 739, 745, 759 ...) and once
inside td_GPU_helper:

    arr[slice_] += factor * td_helper(*args_)
    arr[slice_] += factor * splitting_assistance(*args_, **kwargs)
    result[tuple(view)] += move_tensor_to_cpu_view(outmat)

`factor * X` allocates a full o^2 v^2 temporary -- 9.02 GiB at 920 AO -- and
fills it, before the `+=` reads it again. The allocation is the expensive part:
a fresh array of that size is first-touched page by page.

The fix exploits the fact that X is ours and is consumed inside the expression
that receives it: return an ndarray subclass whose left-multiplication by a
scalar scales IN PLACE and returns itself. No temporary, no allocation, and
`factor == 1` skips the pass entirely. Every one of the thirteen sites is
covered without touching any of them, because they all consume what we return.

Safe only because of that consumption contract -- the same one that already
licenses move_tensor_to_cpu_view. An expression that used X twice would see the
scaling applied once; none does.

  PYBEST_ACCUM=1
"""
from __future__ import annotations

import numpy as np


class ScaleInPlace(np.ndarray):
    """An array that scales itself when a scalar multiplies it on the left."""

    def __rmul__(self, other):
        if np.isscalar(other) and np.isrealobj(self):
            if other == 1:
                return self                      # nothing to do at all
            np.multiply(self, other, out=self)
            return self
        return super().__rmul__(other)

    __mul__ = __rmul__


def install(verbose: bool = True) -> bool:
    try:
        import pybest.linalg._gpu_support as gs
        import pybest.linalg.crosslib_batching as cb
    except ImportError as exc:                              # noqa: BLE001
        if verbose:
            print(f"# accum: not installed ({exc})", flush=True)
        return False

    backend = getattr(gs, "gpu_backend_select", None)
    if backend not in ("cupy", "pytorch"):
        if verbose:
            print(f"# accum: no GPU backend ({backend})", flush=True)
        return False

    stats = {"n": 0}
    orig_get = gs._ops[backend]["get_numpy"]
    orig_as = gs._ops[backend]["as_numpy"]

    def wrap(fn):
        def wrapped(tensor):
            stats["n"] += 1
            out = fn(tensor)
            # .view() on an ndarray is a reinterpretation, not a copy.
            return out.view(ScaleInPlace) if isinstance(out, np.ndarray) else out

        return wrapped

    gs._ops[backend]["get_numpy"] = wrap(orig_get)
    gs._ops[backend]["as_numpy"] = wrap(orig_as)

    # td_GPU_helper and the base.py sites reach the data through these two
    # module-level names, imported at import time, so rebind them too.
    for mod, attr in ((cb, "move_tensor_to_cpu"), (cb, "move_tensor_to_cpu_view")):
        fn = getattr(mod, attr, None)
        if fn is not None:
            setattr(mod, attr, wrap(fn))

    import atexit

    atexit.register(lambda: print(
        f"# accum: scaled in place on {stats['n']} transferred arrays", flush=True))
    if verbose:
        print("# accum: factor * X scales in place, no temporary", flush=True)
    return True
