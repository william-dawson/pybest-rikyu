"""Make RCCSD: unravel cheap, without changing a single number it produces.

Two monkey-patches on DenseFourIndex, both replacing an implementation that does
far more work than the arithmetic needs. Job 161965 measured them at the
dimensions of a 920 AO (H2O)10 CCSD -- nacto 40, nactv 870, t_2 9.02 GiB:

    assign_triu      3.49 s -> 0.49 s   7.1x
    iadd_transpose  10.23 s -> 2.05 s   5.0x
    one unravel     13.7 s  -> 2.5 s    5.4x

and verified every result BITWISE equal, at three sizes, two block sizes.

1. assign_triu (dense_four_index.py:490) calls np.triu_indices(nbasis*nbasis1)
   on every invocation. At 920 AO that is np.triu_indices(34800): two int64
   arrays of 605 M entries, 9.02 GiB of index arrays, followed by a 605 M-element
   fancy-index scatter. But np.triu_indices enumerates the triangle in row-major
   order and row i of the upper triangle is the contiguous slice mat[i, i:], so
   the identical assignment is a sequence of contiguous copies with no index
   arrays and no scatter.

2. iadd_transpose (dense_four_index.py:828) is
   `self.array[:] = self.array + self.array.transpose(t) * factor`, which
   materialises two full temporaries -- 18 GiB at 920 AO -- before the in-place
   store. For an (o,v,o,v) array the permutation (2,3,0,1) sends row (a,b),
   col (c,d) to row (c,d), col (a,b): it is exactly matrix transpose on the
   (ov,ov) view. So it is M += M.T, which blocks in place with only a
   block-sized temporary.

Together these REMOVE about 27 GiB of transient allocation at 920 AO rather than
spending any, which is why they are worth doing on a memory-constrained code.

BLOCK matters: 1024 gives 5.0x on the symmetrisation and 4096 only 1.7x, because
a 1024x1024 double block is 8 MB and stays in cache while a 4096x4096 one is
134 MB and does not. Override with PYBEST_UNRAVEL_BLOCK.

Every case the fast path does not cover -- a shape argument, k != 0, a non-square
reshape, another permutation, an `other` operand, a factor != 1 -- falls through
to PyBEST's own implementation untouched.
"""
from __future__ import annotations

import os

import numpy as np

BLOCK = int(os.environ.get("PYBEST_UNRAVEL_BLOCK", "1024"))


def _triu_rows(mat: np.ndarray, vec: np.ndarray) -> None:
    """Scatter-free upper-triangle assignment, row by contiguous row."""
    n = mat.shape[0]
    off = 0
    for i in range(n):
        ln = n - i
        mat[i, i:] = vec[off : off + ln]
        off += ln


def _symmetrise(mat: np.ndarray, block: int) -> None:
    """M += M.T in place. The result is symmetric, so each off-diagonal block
    pair is written once and mirrored as a transpose."""
    n = mat.shape[0]
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        d = mat[i0:i1, i0:i1]
        d += d.T.copy()
        for j0 in range(i1, n, block):
            j1 = min(j0 + block, n)
            a = mat[i0:i1, j0:j1].copy()
            mat[i0:i1, j0:j1] = a + mat[j0:j1, i0:i1].T
            mat[j0:j1, i0:i1] = mat[i0:i1, j0:j1].T


def install(verbose: bool = True) -> bool:
    """Patch DenseFourIndex in place. Returns False if PyBEST looks different
    from what this was written against, in which case nothing is changed."""
    try:
        from pybest.linalg.dense.dense_four_index import DenseFourIndex
        from pybest.linalg.dense.dense_one_index import DenseOneIndex
    except ImportError as exc:                              # noqa: BLE001
        if verbose:
            print(f"# unravel: not installed ({exc})", flush=True)
        return False

    orig_triu = DenseFourIndex.assign_triu
    orig_iadd = DenseFourIndex.iadd_transpose

    def assign_triu(self, other, begin4=0, end4=None, shape=None, k=0):
        n = self.nbasis * self.nbasis1
        m = self.nbasis2 * self.nbasis3
        arr = other.array if isinstance(other, DenseOneIndex) else other
        if (shape is not None or k != 0 or n != m
                or not isinstance(arr, np.ndarray) or arr.ndim != 1
                or not self.array.flags.c_contiguous):
            return orig_triu(self, other, begin4, end4, shape, k)
        if isinstance(other, DenseOneIndex):
            end4 = other.fix_ends(end4)[0]
        _triu_rows(self.array.reshape(n, m), arr[begin4:end4])
        return None

    def iadd_transpose(self, transpose, other=None, factor=1.0):
        n = self.nbasis * self.nbasis1
        if (other is not None or tuple(transpose) != (2, 3, 0, 1)
                or factor != 1.0 or n != self.nbasis2 * self.nbasis3
                or not self.array.flags.c_contiguous):
            return orig_iadd(self, transpose, other, factor)
        _symmetrise(self.array.reshape(n, n), BLOCK)
        return None

    DenseFourIndex.assign_triu = assign_triu
    DenseFourIndex.iadd_transpose = iadd_transpose
    if verbose:
        print(f"# unravel: assign_triu sliced, iadd_transpose blocked in place "
              f"(block={BLOCK})", flush=True)
    return True
