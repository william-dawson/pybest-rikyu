---
name: pybest-ccsd-cost-model
description: Use when reasoning about the cost of a PyBEST RCCSD calculation - which step dominates, how many FLOPs a term costs, what crosses the host-device link, why a measured time is above or below expectation, or which terms reach the GPU. Covers the full CCSD path step by step with the actual code, an analytic FLOP model per term, and validation against measurements on GB200.
user-invocable: true
---

# PyBEST RCCSD: the path, the code, and an analytic cost model

> **Confidence, as of 2026-10-02 — read before quoting any of this.**
>
> | | status |
> |---|---|
> | The execution path and the term list | **Measured/read** from `cc/rccsd.py:686-860`. Reliable. |
> | Per-term FLOP costs | **Derived** by me from the subscripts, assuming the cheapest contraction order. PyBEST may order differently, and `opt_einsum` may pick another path. Treat as an upper bound on the optimal cost, not as what the code does. |
> | The term-14.4 dominance | **Derived**, but robust: it is the only `v^4` term and the ratio to the `o^3 v^3` terms is large. |
> | Achieved-efficiency figures (74%, 14% of peak) | **MISLEADING, see below.** They divide derived FLOPs by GPU SECTION time, which is mostly not kernel execution. They are not kernel efficiencies. |
> | The `o^2`-skinny-GEMM explanation of that 5x gap | **REFUTED** by Nsight Compute (job 161377). Kernels run at ~98% of compute roofline at both o=100 and o=40. |
> | "~half the GEMM work was not emulated" | **Inference** from combining two models. Weakest claim in this file. |
>
> None of the derived items has been checked against a profiler. An Nsight
> Systems timeline (`cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`,
> `cuda_api_sum`) would settle all of them and is the planned next step. Until
> then, prefer the measured numbers in `results/` over any ratio computed here.

Symbols used throughout:

| | |
|---|---|
| `o` | active occupied orbitals = `occ_model.nacto[0]` (total occupied minus frozen core) |
| `v` | active virtual orbitals = `occ_model.nactv[0]` |
| `x` | Cholesky vectors, `nchol`. Measured ~7.3xN at threshold 1e-5 (not 5xN) |
| `N` | basis functions, `o + v + ncore` |

**Watch `o` carefully.** It is the ACTIVE occupied count. (H2O)10 has 50 occupied
orbitals but with frozen core `o = 40`, and that factor drives the efficiency of
the dominant term (see "Why the synthetic benchmark flatters itself").

## 1. The path, in execution order

```
get_gobasis                     basis set, cheap
compute_cholesky_eri            CPU ONLY (libchol). O(x N^2) storage, large time
RHF                             CPU + some GPU-dispatched contractions
RCCSD.__call__
  +- transform_integrals        4-index AO->MO ("Index Trans"), indextrans= knob
  +- RCCHamiltonianBlocks       Fock + ERI blocks. With Cholesky these are VIEWS
  +- MP2 guess                  initial t_1, t_2
  +- PBQN solver, per iteration:
  |    +- cc_residual_vector    THE COST. singles + doubles residual
  |    +- amplitude update      DIIS / quasi-Newton
  |    +- calculate_energy      cheap, o^2 v^2
  +- (optional) vfunction_l     lambda equations, same shape as the residual
```

Only `cc_residual_vector` matters for scaling. Everything else is setup or
O(o^2 v^2).

**Setup is not negligible and grows faster than you expect.** Measured on GB200,
(H2O)10, CuPy:

| AOs | Cholesky | SCF | setup as % of run |
|---:|---:|---:|---:|
| 240 | 5.7 s | 16.1 s | ~20% |
| 580 | 131 s | 190 s | 24% |
| 920 | 625 s | 667 s | 23% |
| 1150 | 1911 s | 1810 s | **29%** |

Both are CPU-bound regardless of GPU settings, so wall-clock speedups are capped
by Amdahl well before the GPU is the issue. Always read the per-section timer.

## 2. The blocks the residual consumes

`rcc_cache.py` builds these. The comment there matters:

```python
# We keep one whole CholeskyFourIndex to rule them all.
# Non-redundant blocks are accessed as views.
if isinstance(arr, CholeskyFourIndex):
    return (partial(arr.view, **get_range(string)),)
return (partial(arr.copy, **get_range(string)),)   # Dense: real copies
```

| block | dense shape | note |
|---|---|---|
| `eri_oooo` | o^4 | small |
| `eri_ooov`, `eri_oovo` | o^3 v | |
| `eri_oovv`, `eri_ovov` | o^2 v^2 | |
| `eri_vovv` | o v^3 | |
| `eri_vvvv` | **v^4** | NEVER materialised under Cholesky; stays `(x,v,v)` pairs |
| `exchange_*` | as above | `<kl\|cd> - 2<kl\|dc>` combinations |
| `t_1` | o v | |
| `t_2` | o^2 v^2 | stored `(o,v,o,v)` |

`eri_vvvv` being a Cholesky view is the whole reason CCSD is feasible: at
N=1000 a dense `v^4` is ~32 TB.

## 3. The residual, term by term, with costs

From `cc/rccsd.py:686-860`. Cost is multiply-accumulates; double it for FLOPs.
The comment numbers (7.0, 14.4 ...) are PyBEST's own.

### Singles, output `o x v` - all subdominant

| # | code | cost |
|---|---|---|
| 7.0 | `eri_ovov.contract("abcd,cd->ab", t_1, factor=-1)` | o^2 v^2 |
| 7.1 | `eri_oovv.contract("abcd,bd->ac", t_1, factor=2)` | o^2 v^2 |
| 5 | `exchange_oovv.contract("abcd,bc->ad", t_1)` -> `Z_kd` | o^2 v^2 |
| 3 | `exchange_oovv.contract("abcd,ecad->eb", t_2)` -> `mat_oo` | o^3 v^2 |
| 4 | `exchange_oovv.contract("abcd,bead->ec", t_2)` -> `mat_vv` | o^2 v^3 |
| 6.1 | `exchange_ooov.contract("abcd,becd->ae", t_2)` | o^3 v^2 |
| 6.2 | `eri_vovv.contract("abcd,edbc->ea", t_2)` (x2 variants) | o^2 v^3 |

### Doubles, output `o^2 v^2` - where the time goes

| # | code | cost |
|---|---|---|
| 12.0 | `t_2.contract("abcd,ec->edab", mat_oo)` | o^3 v^2 |
| 12.1 | `t_2.contract("abcd,ed->ceab", mat_vv)` | o^2 v^3 |
| 13.0 | `eri_vovv.contract("abcd,ec->bdea", t_1)` | o^2 v^3 |
| 13.5 | `eri_vovv -> intmat`, then `intmat.contract("abcd,cefd->abfe", t_2)` | o^2 v^3 + **o^3 v^3** |
| 14.0 | `eri_oovv.contract("abcd,befd->feac", t_2)` then x `t_2` | **2 o^3 v^3** |
| 14.1 | as 14.0, different permutation | **2 o^3 v^3** |
| 10.4 | in `get_intermediate_w_jibk`: `eri_vovv.contract("abcd,ecfd->efab", t_2)` | **o^3 v^3** |
| 11.3 | in `get_intermediate_u_iakc`: `exchange_oovv.contract("abcd,efbc->efad", t_2)` | **o^3 v^3** |
| 12.3 | `u_ovov.contract("abcd,efcd->abef", t_2)` | **o^3 v^3** |
| 11.3b | `exchange_oovv.contract("abcd,efad->efbc", t_2)` then x `t_x` | **2 o^3 v^3** |
| 13.1 | `eri_oovv.contract("abcd,befd->acfe", t_2, factor=-1)` | **o^3 v^3** |
| 13.2 | `eri_ovov.contract("abcd,cefd->aefb", t_2, factor=-1)` | **o^3 v^3** |
| 13.4 | `eri_vovv -> intmat`, then x `t_2` | o^2 v^3 + **o^3 v^3** |
| 14.2/14.5 | `eri_oovv.contract("abcd,ecfd->efab", t_2)` + `eri_oooo`, then x `t_2` | 2 o^4 v^2 |
| 14.3 | `eri_oovv.contract("abcd->acbd")` | o^2 v^2 copy |
| **14.4** | **`eri_vvvv.contract("abcd,ecfd->eafb", t_2, out=out_d)`** | **v^4 (x + o^2)** |

### The dominant term

Term 14.4 is the particle-particle ladder, PyBEST's own comment calls it
"bottleneck contraction". Under Cholesky `abcd` expands to `xac,xbd`, so the
subscript cuBLAS actually sees is

```
xac,xbd,ecfd->eafb
```

which is entry #1 of `gpu_contraction_optimized`. The cheap evaluation order is

1. `(xac,xbd) -> abcd`   cost **x v^4**
2. `abcd,ecfd -> eafb`   cost **o^2 v^4**

so **v^4 (x + o^2)** in total. Note `x` is usually LARGER than `o^2`: at 1150 AO,
x~8400 against o^2=1600, so **reconstructing the integrals costs 5x more than
contracting them with the amplitudes.** Any work on this term should attack
step 1 first.

The other ~12 doubles terms are all `o^3 v^3`. Their sum relative to 14.4 is
`12 o^3 v^3 / (v^4 (x+o^2))`, which falls as `v` grows - hence the C-split share
of GPU time rising with basis size (measured 13.8 -> 31.1 -> 51.8 -> 58.0% for
CuPy across 240/580/920/1150 AO).

## 4. Validated against measurement

```python
ladder_flop = 2 * v**4 * (x + o**2)
others_flop = 2 * 12 * o**3 * v**3
```

Native FP64 DGEMM peak on one B200, measured: **38.4 TFLOP/s**.

| case | o | v | x | ladder FLOP | measured | achieved | % peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| synthetic N=800 (ladder alone) | 100 | 700 | 4000 | 6.72e15 | 235.2 s | 28.6 TFLOP/s | **74%** |
| molecular 1150 AO, CuPy | 40 | 1100 | 8400 | 2.93e16 | 5289 s | 5.5 TFLOP/s | **14%** |

Same term, same code, same GPU - 5x different efficiency.

### Why those numbers are not kernel efficiencies

`abcd,ecfd->eafb` reshapes to `(ab),(cd) x (ef),(cd) -> (ab),(ef)`, i.e.

```
M = v^2    N = o^2    K = v^2
```

so `N = o^2` is the short dimension and frozen-core molecular runs have small
`o`: the paper's synthetic Table 4 uses nocc=100, where a real (H2O)10 run has
o=40, a 6.25x smaller `N`.

**That shape difference is real but it does NOT cost efficiency.** Nsight
Compute, holding `v` fixed at 300 and varying only `o` (job 161377):

| | GEMM N = o^2 | Compute (SM) Throughput | DRAM Throughput | grid (2nd kernel) |
|---|---:|---:|---:|---:|
| o=100 | 10,000 | 98.36% / 98.46% | 2.6% / 7.8% | 563,200 |
| o=40 | 1,600 | **98.23% / 98.39%** | 2.7% / 5.0% | **112,640** |

The grid shrinks 5x with `o^2`, exactly as the shape predicts, and throughput is
unchanged. Both backends agree to within 1%.

**So the 74%-vs-14% figures were an artifact of the denominator.** They divide a
derived FLOP count by `GPU:` SECTION time, and Nsight Systems (job 161307) shows
section time is overwhelmingly allocation, pinning, transfer and Python: at
240 AO the GPU executes kernels for 3.5 s inside 61.6 s of `GPU: C-split` plus
`GPU: Generic`. Section time is not kernel time, so the ratio was never an
efficiency.

The correct picture, from three independent measurements:

- The GPU does its arithmetic at **~98% of the hardware limit**.
- Those kernels are a **small minority of wall time**.
- Therefore **every remaining opportunity in this code path is host-side.**

That also explains two otherwise puzzling results: FP64 emulation returned
1.15x where a square-DGEMM microbenchmark promised 2.90x, and simply fixing host
memory pinning returned 37% (job 161361).

When predicting a molecular time from a synthetic one, correct for the host-side
work, not for `o^2`. Low occupancy (12-25%) is normal here -- FP64 tensor-core
GEMMs are register- and shared-memory-limited per block -- so ncu's occupancy
advice is not headroom when SM throughput is already 98%.

## 5. Data movement

Per call, operands cross host->device and the result comes back:

| array | bytes |
|---|---|
| `xac`, `xbd` | 8 x v^2 each (times the Cholesky batch count) |
| `ecfd` (`t_2`) | 8 o^2 v^2 |
| result | 8 o^2 v^2 |

Arithmetic intensity of the ladder is roughly
`v^4(x+o^2) / (2 x v^2 + 2 o^2 v^2)` MAC per element, which grows as `v^2` - so
larger bases are MORE compute-dense, not less. Consistent with measurement:
memory-controller utilisation stayed at 18-25% up to N=1200 and only reached
82% at N=1300.

C-split chooses batch counts from a free-VRAM query; see the
`pybest-gpu-offload` skill for the cliff and the drift, both of which change the
number of transfers without changing the FLOP count.

## 6. Caveats learned the hard way

- **Emulation-eligible is not the same as GEMM-bound.** cuBLAS FP64 emulation at
  39 mantissa bits is 2.90x on a large square DGEMM but gave only 1.148x on this
  contraction at N=800. Profiling resolved why: kernels are a small minority of
  the time (3% of wall clock at 240 AO) and already run at 98% of roofline, so
  there is very little for faster arithmetic to win. Emulation is not the lever
  for this code path; host-side memory handling is.
- **`splitting_assistance` RETURNS its result.** It does not fill the array
  passed as the last operand; `base.py` does
  `arr[slice_] += factor * splitting_assistance(...)`.
- **`x` is ~7.3xN at threshold 1e-5**, not the 5xN the synthetic benchmark
  assumes. Measured 1755 at 240 AO, ~8400 at 1150 AO.
- **The timer says "CPU time usage" but reports wall time** (`Timer` uses
  `time.perf_counter()`).
