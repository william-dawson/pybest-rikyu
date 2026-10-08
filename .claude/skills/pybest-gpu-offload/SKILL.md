---
name: pybest-gpu-offload
description: Use when working on PyBEST's GPU path - deciding whether a method or contraction can reach the GPU, choosing CuPy vs PyTorch, interpreting C-split vs X-split, diagnosing a suspected silent CPU fallback, or explaining a performance cliff at large problem size. Covers the dispatch mechanism, which modules qualify, the batching decision, and the known defects.
user-invocable: true
---

# PyBEST's GPU offload: dispatch, batching, and its failure modes

> **Confidence, as of 2026-10-02.** Sections 1-3 are read directly from the
> source and are reliable. Section 4's two defects are **confirmed to exist** in
> the code and their numbers are measured, but that they *cause* the observed
> degradation is **inferred from correlation** with the timing series. Section 5's
> "overhead-bound" reading of PyTorch is a **hypothesis** from coarse counters.
> Nothing here has been checked with a profiler yet.

## 1. One choke point

Offload is not per-method. Everything funnels through
`NIndexObject.contract()` in `src/pybest/linalg/base.py`, dispatching on
`select`:

- `select="td"` forces CPU
- `select="cupy"` / `"pytorch"` forces GPU via `splitting_assistance`
- `select=None` (the normal case) goes to GPU **iff** the subscript is one of the
  18 entries in `_gpu_support.py:gpu_contraction_optimized` AND a backend is
  available
- dense operands with `ndim <= 2` are deliberately skipped

Runtime control is environment-only, read at import:

```
PYBEST_CUPY_AVAIL=1 | PYBEST_PYTORCH_AVAIL=1     # CuPy wins if both
PYBEST_C_SPLITTING=1 xor PYBEST_X_SPLITTING=1    # C if neither or both
```

## 2. Only a Cholesky operand can reach the GPU

All 18 patterns begin `xac,xbd,`. That prefix exists only because

```python
CholeskyFourIndex.einsum_index('abcd')  ->  'xac,xbd'
DenseFourIndex.einsum_index('abcd')     ->  'abcd'
```

So a dense four-index operand can never match, whatever its subscript.
`rcc_cache.py` keeps "one whole CholeskyFourIndex to rule them all" and hands out
views, which is why `from_cache("gvvvv")` stays Cholesky and does reach the GPU.

**Check the receiver, not just the subscript.** A contraction whose dense form
looks eligible will still run on CPU if its receiver came from `init_cache(...)`.

## 3. Which modules actually qualify

Counted by matching call sites against the pattern list, then checking receivers:

| module | GPU-shaped call sites | where they sit |
|---|---:|---|
| `cc` | 31 | **14 in `cc_residual_vector`** - the per-iteration hot path |
| `ip_eom` | 73 | all in one-time `set_hamiltonian_*` setup |
| `ea_eom` | 21 | some in `build_subspace_hamiltonian` - per iteration |
| `ee_eom` | 15 | mixed |
| `geminals` (pCCD) | **0** | pCCD amplitudes are 2-index pairs; it never emits the 4-index ladder |
| `pt`, `sapt` | **0** | |

Consequences:

- **Benchmarking bare pCCD measures a CPU path.** For a pCCD-family method that
  uses the GPU, use `RpCCDLCCSD` (`cc/rlccsd.py` + `rlccsd_base.py`, 10 eligible
  sites).
- **IP-EOM will not speed up** despite having the most call sites - accelerating
  one-time Hamiltonian construction is Amdahl-bounded. EA-EOM is the interesting
  EOM case, with eligible calls inside the iteration and operating on `gvvvv`.
- 5 of the 18 patterns are emitted by no module at all
  (`abcd,cedf->abfe`, `abcd,cedf->aefb`, `abcd,edfc->eafb`, `abcd,efcd->efab`,
  `abcd,efcd->efba`): dead optimisation, or a naming drift worth reporting.

## 4. The batching decision, and two defects in it

`c_splitting` (crosslib_batching.py:592-1166) has more than one internal path.
For `xac,xbd,ecfd->efab` the one that runs is at line 702, NOT the
`get_batch_sizes` path at 1082 - confirmed by instrumenting both.

```python
memhave = memory_usage() * 0.98            # DRIVER-level free VRAM

if args[2].nbytes < memhave * 0.4:         # args[2] is ecfd
    # cheap: split axes a and b only
else:
    # "if efcd is more than 45% of memory we have to split third axis (c)"
```

### Defect 1: a performance cliff keyed to a ratio

`ecfd` is `8 o^2 v^2` bytes. The test flips when that passes `0.4 x memhave`.
With a cold card (~183 GiB free) the crossing is at **N ~ 1082** for nocc=100,
nvec=5N - which is between the published N=1000 and N=1100 columns, and where
both backends lose performance. Past it, C-split splits a third axis and issues
many more, smaller batches.

The branch and the arithmetic are certain; **that it causes the degradation is
inferred from the coincidence of position.** We have not measured batch counts on
either side of the crossing, which would confirm or kill it cheaply.

### Defect 2: the reading drifts, so the same call gets slower

`memory_usage()` is `memGetInfo()[0]` (CuPy) or `torch.cuda.mem_get_info()[0]`
(PyTorch) - driver-level free VRAM, which excludes whatever the library's
caching allocator holds. It is queried per call, so it falls as the run proceeds:

| N | pass 1 | pass 2 | CuPy pass-2 penalty | PyTorch pass-2 penalty |
|---:|---:|---:|---:|---:|
| 1200 | 183.3 GiB | 138.1 GiB | +5.7% | +18.0% |
| 1300 | 183.3 GiB | 129.6 GiB | +9.1% | +15.9% |

Identical input, slower run, because the budget shrank and the code split harder.
In a converged calculation every iteration after the first pays this.

**Both libraries drift by the same ~45-54 GiB**, so the drift explains the common
degradation but NOT the backend difference. A plausible fix for the authors:
call `clean_memory()` immediately before the query, or size against the
allocator's own accounting rather than the driver's.

### How much the drift costs, and a hypothesis that was wrong

Freezing the first (cold-card, 183.30 GiB) reading for the whole process:

| | baseline | frozen | change |
|---|---:|---:|---:|
| ladder N=800, PyTorch | 288.42 s (warmup) | 278.03 s | -3.6% |
| ladder N=1200, PyTorch | 1948.04 s | 1704.41 s | **-12.5%** |
| ladder N=1200, CuPy | 1622.32 s | 1614.24 s | -0.5% |

**I first read that as removing query overhead. That was wrong, and the source
settles it in one line:** `free_memory` is `lambda: torch_cuda.mem_get_info()[0]`
(`_gpu_support.py:293`), a bare `cudaMemGetInfo`, and `crosslib_batching.py`
calls `memory_usage()` five times -- lines 241, 503, 702, 1079 and one more.
Five microsecond calls cannot be 12.5% of a 1948-second run.

So the entire gain is the **batch plan changing**, and the number measures
something more interesting than a tuning knob: it is how much performance the
batching heuristic leaves on the table at N=1200 by sizing against a budget that
has silently shrunk. It grows with N (-3.6% at 800, -12.5% at 1200), which is
what a plan-sensitivity effect should do and what a fixed syscall cost could not.

**Do not ship freezing as a fix.** It makes the code allocate against a stale
optimistic estimate: the frozen value is the largest the run will ever see, so
PyBEST stays permanently more willing to attempt large batches than its own
memory accounting would allow. On a code whose entire reason for batching is to
not exceed memory, trading estimate reliability for 12.5% is the wrong trade.
The fix worth proposing upstream is making `memhave` *correct* -- adding the
allocator's own cached-but-free blocks, which are genuinely available to the next
allocation and which the driver-level query excludes by construction.

### Overhead-reduction ideas, and why three of four are not weapons

Once pinning fixed the D2H bandwidth, these were the candidates. Recording the
arithmetic because each one looks attractive until it is costed:

1. **Non-blocking D2H with event-based sync** instead of a device-wide
   `synchronize()` per transfer. Real, but it **costs memory**: a single reused
   staging buffer cannot be refilled until the previous copy retires, so genuine
   overlap needs double buffering, and the pinning fix's whole appeal is that it
   costs nothing.
2. **Eliminating the 85x H2D re-transmission** (260 GB uploaded for a 0.77 GB
   ERI) with a VRAM-capped cache of the Cholesky vectors. **Fails on its own
   arithmetic:** H2D already runs at 128 GB/s, so those 260 GB cost 2.03 s of a
   115.7 s run -- under 2%, below run-to-run noise. It is only worth revisiting
   if the *host-side* cost of issuing 1211 uploads is large, and roughly 18 s of
   the 24.7 s remaining in `GPU: Generic` after pinning is still unattributed.
   That is answerable from the stored `.nsys-rep` files at zero GPU cost, and
   should be answered before any VRAM is spent.
3. **Fewer `clean_memory()` calls** (40 call sites, many in loops;
   `cudaFree` + `cudaMalloc` are ~7 s of 115 s). **Rejected on risk:** fewer
   flushes means more memory held, which changes what `mem_get_info` reports,
   which changes the batch plan -- the same entanglement as Defect 2. Reliable
   memory estimates matter more than 6%.
4. **`RCCSD: unravel`**, 462 s at 920 AO, reproducing to within 1% between
   backends, so pure host work with no GPU involvement -- about 12% of the run
   and completely unexamined. Agreed as worthwhile, but a substantial change
   rather than a patch.

## 5. CuPy vs PyTorch: the answer depends on what you measure

| | synthetic ladder, N=1200-1300 | molecular CCSD, 1150 AO |
|---|---|---|
| winner | **CuPy** by 20-25% | **PyTorch** by 21% |
| vs H100 | CuPy 1.88-2.01x; **PyTorch 0.92-0.96x, i.e. slower than Hopper** | - |
| `GPU: Generic` | - | CuPy 3836 s vs PyTorch 1668 s (**CuPy 2.3x slower**) |
| SCF | - | CuPy 1810 s vs PyTorch 176 s (**CuPy 10.3x slower**) |

The synthetic benchmark isolates the C-split term, where CuPy is genuinely
stronger. Real CCSD also pays for the generic path and SCF, where CuPy is much
weaker - and Table 4 cannot see either. **The synthetic benchmark recommends the
opposite backend from the real workload.**

At N=1300 the memory controller reaches **82% for CuPy and 28% for PyTorch**,
while both report 100% "GPU utilisation". A kernel saturating neither arithmetic nor
bandwidth is *plausibly* overhead-bound, which would also explain PyTorch paying
~2x more for finer splitting - but `utilization.gpu` merely reports that a kernel
was resident, and 30-second sampling can miss bursts. **Treat this as the leading
hypothesis, not a finding.** `cuda_api_sum` launch counts would settle it.

The CuPy SCF penalty reproduces at 9.7x / 11.1x / 10.3x across 580 / 920 / 1150
AO. `SCF` own time is ~5 s in both; the difference is entirely in children that
route through the GPU contraction path. The paper never examines SCF.

## 6. Silent CPU fallback - the main methodological hazard

Every GPU path degrades to CPU **without failing**. `select=None` wraps GPU calls
in a bare `except Exception`, and `MemoryError` falls back to `td_helper` with a
warning emitted only at `log.do_high`. A "GPU run" can produce perfectly correct
numbers having never touched the GPU.

Never infer GPU use from the environment variable. Corroborate with:

1. `log.level = log.high` - **NOT** `log.set_level()`, which does not exist, and
   only AFTER SCF, because PyBEST 2.2.0 crashes at that verbosity during DIIS
   (`scf_diis.py:480` guards only `EmptyData` around
   `min(state.energy for state in ...)` while the energies are still `None`).
2. Sampling `nvidia-smi --query-gpu=utilization.gpu,utilization.memory`. Add
   `utilization.memory`: it separates "busy computing" from "busy moving data",
   which plain GPU utilisation cannot.
3. The per-section timer, which splits `GPU: C-split` from `GPU: Generic`.

## 7. `clean_memory()` is the largest single cost at scale

Measured at 920 AO, two CC iterations, fully patched (job 162831, nsys and
cProfile independently):

| | time | calls |
|---|---:|---:|
| GPU kernels (all cutlass d884gemm) | 1002.9 s | 1346 |
| **`cudaFree`** | **1049.4 s** | 3638 |
| `cudaMalloc` | 90.5 s | 3641 |
| H2D, 8.50 TB | 61.5 s | 1383 |
| D2H, 788 GB | 4.1 s | 824 |

cProfile attributes it to one built-in: `torch.cuda.empty_cache()`, **1059.8 s
over 2399 calls at 0.44 s each**, reached from `_gpu_support.clean_memory()`,
which `crosslib_batching` calls from ~40 sites, many inside loops. It is 99% of
`c_splitting`'s cumulative time.

Meanwhile the kernels execute 3.16e16 FLOP in 1002.9 s = 31.5 TFLOP/s, **82% of
the measured 38.4 TFLOP/s DGEMM peak**. The arithmetic is near-optimal; the GPU
is simply idle for half the run while the host is inside `cudaFree`.

### Why the flush is there

`memory_usage()` is `torch.cuda.mem_get_info()[0]` -- **driver-level** free
VRAM, which by construction excludes every block the caching allocator holds,
including blocks that are free and immediately reusable. That is why the reading
drifts 183 -> 138 GiB through a run (Defect 2 above). Flushing the allocator is
what makes the reading true again. PyBEST is paying 1049 s to work around an
undercount in its own accounting.

### The fix to propose, and its risk

Report `driver_free + (torch.cuda.memory_reserved() - memory_allocated())` and
the flushes become unnecessary. This makes the estimate *more* accurate, not
more optimistic -- unlike freezing the reading, which was measured at +12.5% and
correctly rejected because it sizes batches against the largest value a run ever
sees.

The risk to test is fragmentation: cached blocks may be the wrong shapes to
serve one large request, so a run that never flushes could OOM where flushing
succeeded. Keep a flush on the `MemoryError` retry path.

### Do not generalise from 240 AO

Everything above inverts the 240 AO picture, where kernels are 3.3 s of 66.6 s
(5%) and the host-side remainder is 78%. At that size `GPU: C-split` is 3.5 s of
70 s; at 920 AO it is 1782 s of 2946. **Any conclusion about where time goes
must be re-measured at the size being discussed.**

### Correction: the flush costs 7%, not 56%

Job 163891, 920 AO, maxiter=2, three cells paired:

| mode | CCSD | vs control | flushes skipped |
|---|---:|---:|---:|
| flush (control) | 1818.24 s | -- | 0 |
| corrected accounting only | 1822.59 s | +0.2% | 0 |
| no flushes | **1683.24 s** | **-7.4%** | 2031 |

Energy identical, peak RSS flat, no OOM at this size.

So the 1049 s that nsys attributes to `cudaFree` is **about 85% kernel-drain
wait**: `cudaFree` synchronises, so the host blocks there while the device
finishes. Removing 2031 flushes recovers 135 s, 66 ms per flush against the
288 ms attributed. The claim above that "freeing memory costs as much as the
arithmetic" is WRONG as stated and is retracted; what is true is that the GPU
computes for only ~53% of the run, and the allocator is not the reason.

Correcting the estimate on its own changes nothing, because corrected and raw
agree (183.1 GiB both) at the point the batch plan is chosen -- the drift
appears later in a pass. So the two ideas are separable and only one pays.

**Methodological note.** This is the second time a profiler attribution was
read as cost when it was waiting: the first was treating an emulation `G + O`
fraction as a GEMM fraction. For any blocking CUDA API -- `cudaFree`,
`cudaMemcpy` to pageable, `cudaDeviceSynchronize` -- the attributed time is an
upper bound on the work, and the only way to separate it is to remove the call
and measure.

### Why `noflush` is not shipped: it costs VRAM, not nothing

Skipping `clean_memory()` is worth **−5.9% at 580 AO and −7.4% at 920** and
changes no result. It is still not in the shipped stack, because sampling GPU0
every 20 s through job 163891 at 920 AO shows what it actually buys that with:

| mode | peak VRAM | mean VRAM |
|---|---:|---:|
| flush (stock) | 119.5 GiB | 30.1 GiB |
| corrected accounting only | 119.5 GiB | 30.4 GiB |
| **no flush** | **146.8 GiB** | **94.8 GiB** |

Peak rises 27 GiB but the **mean rises 3.1x**, 30 to 95 GiB, on a 184 GiB card.
Nothing bounds it: the allocator simply never gives anything back.

Scaling to 1150 AO, where `o^2 v^2` grows 9.02 -> 15.5 GiB, projects a peak
around 250 GiB against 184 available. So the measured win at 580 and 920 AO is
borrowed against a failure at the next size up, and on a code whose entire
batching machinery exists to not exceed memory that is the wrong trade.

**The idea is not dead, and the diagnosis points at the fix.** The flush is
indiscriminate rather than useless: PyBEST calls it from ~40 sites regardless
of whether memory is tight. Flushing only when memory is actually scarce --
`if corrected_free < watermark * total: clean_memory()`, using the corrected
accounting (driver free plus the allocator's cached-but-free blocks) -- should
keep most of the gain with a hard bound on growth. Untested; worth doing.

Note also that the corrected accounting on its own is a null (+0.2%), because
corrected and raw agree at the point the batch plan is chosen -- 183.1 GiB
both. The two ideas are separable and only the flushing one pays.

## 8. Things we tried that did not work

Kept here so nobody spends the GPU hours twice. Percentages are complete RCCSD
runs, paired inside one job, PyTorch unless stated.

| change | result | why |
|---|---|---|
| The pinned-buffer fix applied to **CuPy** | +7.5% at 240 AO, +5.2% at 580 | CuPy already stages through a pinned pool and reaches 126 GB/s unaided; interposing on it only adds a copy |
| **Freeze** the free-VRAM reading | −12.5% at N=1200, rejected | Real, but it is the batch plan changing, not overhead removed. Sizes batches against the largest reading a run ever sees, on a code whose batching exists to not exceed memory |
| **Correct** the free-VRAM accounting | +0.2%, null | Corrected and raw agree (183.1 GiB both) at the moment the plan is chosen; the drift appears later in a pass |
| **Skip the allocator flush** | −5.9% / −7.4%, not shipped | See section 7: mean VRAM 30 → 95 GiB |
| Scale `factor * X` **in place** | +0.5%, null | Identical traffic, 3.94 TB either way. The temporary is the same size every call, so numpy reuses the arena block and there are no repeat page faults to save |
| **Chunked** threaded accumulate | −12.5% and **wrong** | Energy moved 3.9e-4 Ha. The fallback scaled every operand that was not literally `inputs[0]`, so `np.add(self, other)` scaled `other` too |
| Override the batch count (`parts=`) | +9% to +139% | 2 runs out of memory, 4 is 9–10% slower, 8 is 28% slower for CuPy and 139% for PyTorch. The adaptive heuristic earns its complexity |
| `OMP_NUM_THREADS` | flat | Only the Cholesky decomposition scales (136 s at 64 threads, 425 s at 8). CCSD-side CPU work does not |
| **SYRK** for the reconstruction | not attempted | The reconstructed `(vv|vv)` block is a Gram matrix, so `dsyrk` would halve its FLOPs — but the kernels already run at 82% of DGEMM peak, so this optimises what is not the constraint |
| **FP64 emulation** | 1.08–1.15x inside PyBEST | Measured at nocc=100, where reconstruction is 29% of the ladder against 84% in a real large-basis case — the wrong regime. See the benchmarking skill |

### The recurring trap

Four of these were defensible on paper and wrong in practice, and three share a
root: **a profiler's attributed time is an upper bound on removable work, not
an estimate of it.**

- `cudaFree` showed 1049 s against 1003 s of kernels. Removing 2031 flushes
  recovered 135 s; the rest was the host blocking while the device drained.
- The emulation `G + O` fraction was read as a GEMM fraction.
- The accumulate's cost was attributed to allocation; it is the memory traffic.

For any blocking CUDA API — `cudaFree`, `cudaMemcpy` to pageable,
`cudaDeviceSynchronize` — remove the call and measure rather than reasoning
about the attribution.

### And the operational one

Three times a patch silently failed to apply on the cluster, because a driver
had been regenerated locally and not re-uploaded. A stale upload does not
error, it produces a plausible number. Every job that varies a patch must
assert that the patch announced itself, per cell. Equally, a guard that has
never been seen to fire on a known-bad input has not been tested: the energy
check in one job used `grep -aq "$REF"` with a reference beginning in `-`,
which grep parsed as an option bundle, so it reported WRONG on every cell
including correct ones. It needs `grep -aqF --`.
