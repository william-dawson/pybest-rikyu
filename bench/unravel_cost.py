#!/usr/bin/env python3
"""Where does RCCSD: unravel spend its time, and can it be made cheap?

unravel is 462 s at 920 AO -- about 12% of the run, reproducing to within 1%
between CuPy and PyTorch because it is pure host work with no GPU involvement.
After the pinning fix removed the transfer cost it is the largest remaining
addressable item (`cudaMalloc`/`cudaFree` churn is bigger per profile, but
reducing it perturbs the batching decision; this does not perturb anything).

rccsd.py:648 does three heavy things to a t_2 of shape (nacto, nactv, nacto,
nactv) -- 9.69 GB at 920 AO:

    t_2.assign_triu(vector, begin4=nov)
    t_p = t_2.contract("abab->ab")
    t_2.iadd_transpose((2, 3, 0, 1))

and two of them are more expensive than the arithmetic requires:

assign_triu (dense_four_index.py:490) calls np.triu_indices(nacto*nactv). At
920 AO that is np.triu_indices(34800): two int64 arrays of 605 M entries, 9.69
GB of INDEX arrays, built on every call, followed by a 605 M-element fancy-index
scatter. But the upper triangle of a matrix is row-contiguous -- row i is
M[i, i:] -- and np.triu_indices enumerates it in exactly that order, so the same
assignment is a sequence of contiguous slice copies with no index arrays at all.

iadd_transpose (dense_four_index.py:828) is
`self.array[:] = self.array + self.array.transpose(t) * factor`, which
materialises TWO full temporaries before the in-place store. And for an
(o,v,o,v) array the permutation (2,3,0,1) maps row (a,b), col (c,d) to row
(c,d), col (a,b) -- it is exactly matrix transpose on the (ov, ov) view. So it is
M += M.T, which blocks in place with only a block-sized temporary.

This times both forms and checks they agree bitwise. Pure CPU; no GPU is used.

  apptainer exec -B /data1 pybest-rikyu-v3.sif python unravel_cost.py
"""
from __future__ import annotations

import gc
import json
import os
import time

import numpy as np

# (label, nacto, nactv) matching the molecular cases we have run.
# nacto = nocc - ncore, nactv = nbasis - nocc.
CASES = [
    ("240 AO  cc-pVDZ",     40, 190),
    ("580 AO  cc-pVTZ",     40, 530),
    ("920 AO  aug-cc-pVTZ", 40, 870),
]
BLOCK = int(os.environ.get("UC_BLOCK", "2048"))   # rows per block, ov units


def triu_slow(mat: np.ndarray, vec: np.ndarray) -> None:
    """What PyBEST does: build the index pair, then scatter."""
    n = mat.shape[0]
    idx = np.triu_indices(n, 0)
    mat[idx] = vec


def triu_fast(mat: np.ndarray, vec: np.ndarray) -> None:
    """Row i of the upper triangle is mat[i, i:], and np.triu_indices
    enumerates the triangle in that same row-major order, so the packed vector
    can be consumed by contiguous slices with no index arrays."""
    n = mat.shape[0]
    off = 0
    for i in range(n):
        ln = n - i
        mat[i, i:] = vec[off : off + ln]
        off += ln


def symm_slow(arr: np.ndarray) -> None:
    """What PyBEST does: A + A.transpose(2,3,0,1), two full temporaries."""
    arr[:] = arr + arr.transpose(2, 3, 0, 1)


def symm_fast(mat: np.ndarray, block: int = BLOCK) -> None:
    """M += M.T in place, blocked. Only a block-sized temporary, and the
    result is symmetric so the lower half is written as a transpose view."""
    n = mat.shape[0]
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        # diagonal block: symmetrise in place
        d = mat[i0:i1, i0:i1]
        d += d.T.copy()
        for j0 in range(i1, n, block):
            j1 = min(j0 + block, n)
            a = mat[i0:i1, j0:j1].copy()
            mat[i0:i1, j0:j1] = a + mat[j0:j1, i0:i1].T
            mat[j0:j1, i0:i1] = mat[i0:i1, j0:j1].T


def timeit(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def run_case(label: str, o: int, v: int) -> dict:
    ov = o * v
    npack = ov * (ov + 1) // 2
    gib = 8 * ov * ov / 2**30
    print(f"\n{'=' * 70}\n{label}: nacto={o} nactv={v}  ov={ov}", flush=True)
    print(f"  t_2 {gib:.2f} GiB   packed vector {8 * npack / 2**30:.2f} GiB"
          f"   triu_indices would be {2 * 8 * npack / 2**30:.2f} GiB", flush=True)

    rng = np.random.default_rng(0)
    vec = rng.standard_normal(npack)
    res: dict = {"nacto": o, "nactv": v, "ov": ov, "t2_gib": gib,
                 "triu_index_gib": 2 * 8 * npack / 2**30}

    # ---- unpack ----
    a = np.zeros((ov, ov))
    res["triu_slow_sec"] = timeit(lambda: triu_slow(a, vec))
    b = np.zeros((ov, ov))
    res["triu_fast_sec"] = timeit(lambda: triu_fast(b, vec))
    res["triu_equal"] = bool(np.array_equal(a, b))
    res["triu_speedup"] = res["triu_slow_sec"] / res["triu_fast_sec"]
    print(f"  assign_triu   PyBEST {res['triu_slow_sec']:8.2f} s"
          f"   sliced {res['triu_fast_sec']:8.2f} s"
          f"   {res['triu_speedup']:5.1f}x   equal={res['triu_equal']}", flush=True)

    # ---- the diagonal extraction, for scale (not optimised here) ----
    a4 = a.reshape(o, v, o, v)
    res["diag_sec"] = timeit(lambda: np.einsum("abab->ab", a4).copy())
    print(f"  abab->ab      {res['diag_sec']:8.2f} s", flush=True)

    # ---- symmetrise ----
    ref = a.copy()
    res["symm_slow_sec"] = timeit(lambda: symm_slow(a.reshape(o, v, o, v)))
    c = ref.copy()
    res["symm_fast_sec"] = timeit(lambda: symm_fast(c))
    res["symm_equal"] = bool(np.array_equal(a, c))
    res["symm_speedup"] = res["symm_slow_sec"] / res["symm_fast_sec"]
    print(f"  iadd_transpose PyBEST {res['symm_slow_sec']:7.2f} s"
          f"   blocked {res['symm_fast_sec']:7.2f} s"
          f"   {res['symm_speedup']:5.1f}x   equal={res['symm_equal']}", flush=True)

    # ---- the gc.collect() unravel calls on every invocation ----
    res["gc_sec"] = timeit(gc.collect)
    print(f"  gc.collect()  {res['gc_sec']:8.2f} s", flush=True)

    slow = res["triu_slow_sec"] + res["symm_slow_sec"] + res["diag_sec"] + res["gc_sec"]
    fast = res["triu_fast_sec"] + res["symm_fast_sec"] + res["diag_sec"] + res["gc_sec"]
    res["unravel_slow_sec"], res["unravel_fast_sec"] = slow, fast
    res["unravel_speedup"] = slow / fast
    print(f"  -> one unravel {slow:.1f} s -> {fast:.1f} s  ({slow / fast:.1f}x)",
          flush=True)

    del a, b, c, ref, a4
    gc.collect()
    return res


def main() -> int:
    print(f"numpy {np.__version__}  threads: see OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    out = {"block": BLOCK, "cases": {}}
    for label, o, v in CASES:
        try:
            out["cases"][label] = run_case(label, o, v)
        except MemoryError as exc:
            out["cases"][label] = {"error": f"MemoryError: {exc}"}
            print(f"  MemoryError -- skipped", flush=True)
    dest = os.environ.get("BENCH_OUT")
    if dest:
        with open(dest, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {dest}")
    print("\n`equal=True` is the point: these are the same arrays, not an "
          "approximation.\nMeasured against PyBEST's 462 s of unravel at 920 AO "
          "over 4 iterations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
