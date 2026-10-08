# PyBEST on RIKYU — GPU benchmarking brief

Benchmark **PyBEST v2.2.0** on **RIKYU** (RIKEN AI4S, GB200). The focus is the
**GPU offload path**: how PyBEST's Cholesky tensor contractions behave on
Blackwell, first by **reproducing published Hopper/Grace-Hopper results**, then
by extending past what that hardware could reach.

- Docs: <https://fizyka.umk.pl/~pybest/pybest-v2.2.0/index.html>
- Source: <https://www.fizyka.umk.pl/~pybest/downloads/pybest.v2.2.0.tar.gz> (9.8 MB)
- Workspace on RIKYU: `/data1/rkp00012/rku00036/pybest/`

> **NOTE:** this folder lives on the Desktop because macOS TCC blocked
> `~/Documents` mid-session. The original working dir was
> `~/Documents/AI4S/pbest`. If access is restored, either is fine — keep one.

## Status

- [x] Container **built and fully verified**: `pybest-rikyu-v3.sif` (4.8 GB)
- [x] libint2 + libchol compiled; Cholesky ERI enabled; 18 GPU patterns present
- [x] CuPy **and** PyTorch both working on GB200, both doing real FP64 GEMM
- [x] Prior art located and its numbers extracted (see below)
- [x] MCP servers + rikyu skills installed in this folder
- [x] **Table 4 reproduced, N=800-1100**, both backends (jobs 147057, 147127)
- [x] **Molecular CCSD at 240, 580, 920 AO**, both backends (147258, 147264,
      147348, 147598)
- [x] **cuBLAS FP64/FP32 emulation characterised** (147643) and measured inside
      PyBEST (147646)
- [x] **Profiled at 240 AO** (161307, 161952) **and at 920 AO** (162831, nsys
      + cProfile). **The 240 AO balance is NOT representative and earlier
      conclusions drawn from it were wrong.** At 920 AO, two CC iterations,
      fully patched: GPU kernels 1002.9 s executing 3.16e16 FLOP = 31.5 TFLOP/s
      = **82% of the 38.4 TFLOP/s DGEMM peak**; `cudaFree` **1049.4 s** over
      3638 calls; cudaMalloc 90.5 s; H2D 8.50 TB at 138 GB/s = 61.5 s; D2H
      788 GB at 193 GB/s = 4.1 s. cProfile names the culprit exactly:
      `torch.cuda.empty_cache()` **1059.8 s over 2399 calls**, 0.44 s each,
      from `clean_memory()`, which is 99% of `c_splitting`'s time.
      **So the arithmetic is near-optimal and freeing memory costs as much as
      it.** The old "kernels are 5% of the run" figure was a 240 AO artifact,
      where kernels are 3.3 s of 66.6 s; do not carry it forward.
- [x] **Two speedups found, both free** -- pinned host buffer and a cheap
      `unravel`. See "Tuning results" below.
- [x] **Transfer mechanisms priced** (161947): pageable D2H 25-40 GB/s, our
      staged copy 10.8, pinned-without-copy 193, `cudaHostRegister`ed
      destination 192.8, H2D registered 207, and **system-allocated (ATS)
      memory works at 209 GB/s**. Registration costs 0.03 s/GiB once.
- [ ] **Remove the `clean_memory()` cost** -- the single biggest remaining
      item, 57% of a 920 AO CCSD. It exists to correct an undercount:
      `mem_get_info` reports driver-level free VRAM and excludes the caching
      allocator's cached-but-free blocks. Report driver-free PLUS
      `reserved - allocated` and the flush becomes unnecessary. NOT YET TESTED;
      the risk to check is fragmentation causing OOM where flushing prevented it.
- [ ] Accuracy of emulated arithmetic (norms). **Deprioritised**: emulation
      multiplies the GEMM, which is now shown to be 82% efficient and roughly
      half the run, and the existing measurement was taken at nocc=100 where
      reconstruction is 29% of the ladder against 84% in a real large-basis
      case. Numbers moved to `.claude/skills/pybest-rikyu-benchmarking`.
- [ ] pCCD-LCCSD (`RpCCDLCCSD`) -- the pCCD-family method that actually uses
      the GPU path
- [x] Extended past their memory ceiling: N=1200 and N=1300 measured, both
      backends, which their GH200 reported as `n.c.`

## Results

All C-split, `--gpus=2` (64 cores). GPU use corroborated per cell by sampling
`nvidia-smi`, never inferred from the env var.

### Table 4, synthetic ladder `xac,xbd,ecfd->efab` (nocc=100, nvec=5N)

| N | GH200 CuPy | ours | | GH200 PyTorch | ours | |
|---:|---:|---:|---:|---:|---:|---:|
| 800 | 306.6 | **235.18** | 1.30x | 355.9 | **267.78** | 1.33x |
| 900 | 495.0 | **397.97** | 1.24x | 587.1 | **436.25** | 1.35x |
| 1000 | 829.1 | **680.98** | 1.22x | 805.8 | **684.50** | 1.18x |
| 1100 | 1282.7 | **1051.48** | 1.22x | 1401.7 | **1379.94** | 1.02x |

CuPy settles at 1.22x. **PyTorch collapses to 1.02x at N=1100** and is then 24%
slower than CuPy, restoring the ordering their GH200 data shows at that size.

Peak host RSS is **not** a fixed multiple of the operands. CuPy excess is
+70.5, +74.6, +70.6 GiB at N=800/900/1000 and then **+139.2 GiB at N=1100**,
where the splitting evidently changes strategy. Using the ratio measured at the
boundary (1.60) their N=1200 cell needs ~497 GB against a 480 GB machine, and
N=1100 ~397 GB -- which is exactly where their `n.c.` boundary falls.

### Molecular CCSD, (H2O)10, per iteration

| AOs | basis | GH200 CuPy | ours | GH200 PyTorch | ours |
|---:|---|---:|---:|---:|---:|
| 240 | cc-pVDZ | 23.9 s | 20.4 s | 25.4 s | 23.6 s |
| 580 | cc-pVTZ | 5.5 m | 4.12 m | 5.7 m | **3.61 m** |
| 920 | aug-cc-pVTZ | -- | 18.1 m | -- | **15.5 m** |

920 AO is beyond anything they report, so it has no reference; 240 and 580 are
what license it. PyTorch wins at 580 and 920, CuPy at 240.

**The C-split share of GPU time rises steeply with basis size** -- the single
most useful number we have, because it says how much of a real calculation the
contraction they optimised actually governs:

| AOs | CuPy | PyTorch |
|---:|---:|---:|
| 240 | 13.8% | 4.2% |
| 580 | 31.1% | 40.7% |
| 920 | **51.8%** | **65.6%** |

CPU-side costs are large and reproduce to <1% between backends: at 920 AO,
`RCCSD: unravel` 462-466 s, `Ints: CD-ERI` 625-627 s, `Base: contract` own
313-316 s.

**SCF is 10x slower under CuPy than PyTorch** for identical work -- 190 s vs
19.6 s at 580 AO, 667 s vs 60 s at 920 AO. `SCF` own time is ~5 s in both, so
the difference is entirely in children, which route through the GPU contraction
path. Unexplained; the paper never examines SCF.

### Emulated arithmetic -- and why it does not help much

cuBLAS 13.0u2+ emulates GEMM two ways, both selected by environment variable
only, so **PyBEST needs no changes**: fixed-point Ozaki-I/II for FP64, and
BF16x9 for FP32. (BF16x9 is FP32 emulation; it is *not* the FP64 mechanism.)

On square 8192^3 DGEMM (job 147643), against native FP64 at 38.4 TFLOP/s:
ADP 1.32x, 55 bits **1.84x**, 39 bits 2.90x, 31 bits 3.78x; FP32 BF16x9 4.98x.
At 55 bits it is faster *and* 3x more accurate than native FP64.

Inside PyBEST at N=800 (job 147646) almost all of that disappears:

| mode | raw DGEMM | CuPy | PyTorch |
|---|---:|---:|---:|
| ADP | 1.32x | 1.055x | 1.064x |
| 55 bits | 1.84x | 1.106x | 1.085x |
| 39 bits | 2.90x | 1.148x | 1.155x |

Solving `G + O` for the GEMM fraction gives **the same answer from all three
modes**: the matrix multiply is only ~47-50 s of a 235.76 s contraction, about
**20%**. The other 80% is transfer, batching, reshaping and CPU orchestration.

That also explains the modest 1.22-1.30x Blackwell-over-Hopper result: a
bandwidth-bound kernel benefits from GB200's HBM and C2C, not from FP64 compute,
where B200 is not far ahead of H100. **This contraction is data-movement bound.**
Faster arithmetic attacks 20% of the cost; GPU residency would attack the 80%.

Note the corollary for the multi-GPU question: PyBEST is single-GPU, and GPU 1
has sat at 0% and ~9 MiB in every job. We request 2 GPUs only to buy 64 cores
and host memory, on a machine billed per GPU requested.

## What is measured vs what is inferred

Treat this distinction as load-bearing; several attractive explanations here are
not yet demonstrated, and the skills in `.claude/skills/pybest-*` carry the same
caveats.

**Measured, trust it:** every timing and peak-RSS figure; the free-VRAM readings
and their drift; memory-controller utilisation; the emulation microbenchmark;
energies agreeing between backends; reproducibility across nodes and days.

**Read from source, trust it:** the dispatch rule and the 18 patterns; that only
a `CholeskyFourIndex` operand can match them; the per-module eligible-call
counts; the `ecfd < 0.4 * memhave` branch and the formulae around it; the
four PyBEST bugs below.

**Derived, provisional:** the per-term FLOP model and every efficiency figure
computed from it (74% of peak synthetic, 14% molecular).

**Hypotheses, not findings:** that the `0.4 * memhave` branch *causes* the
observed degradation (inferred from position); that the skinny `N = o^2` GEMM
explains the synthetic-vs-molecular efficiency gap; that PyTorch is
overhead-bound; that cuBLAS declined to emulate about half the GEMM work.

**Retracted:** that caching the free-VRAM query removes measurable overhead.
`free_memory` is `lambda: torch_cuda.mem_get_info()[0]` (`_gpu_support.py:293`)
and `crosslib_batching` calls it five times; five microsecond calls cannot be
12.5% of a 1948-second run. The gain was entirely the **batch plan** changing,
which makes it a finding about the heuristic, not a knob to turn -- freezing the
reading sizes batches against the largest value the run will ever see, on a code
whose reason for batching is to not exceed memory. Details in
`.claude/skills/pybest-gpu-offload`.

**Retracted:** that the synthetic contraction is data-movement bound. The FLOP
model puts GEMMs at ~74% of it and memory-controller utilisation of 18-25% at
N<=1200 agrees it is compute-bound. The earlier reading came from treating an
emulation `G+O` fraction as a GEMM fraction.

### Tuning results -- two free speedups

All paired **inside one job**; cross-job drift reaches 8.8% on the ladder (1.2%
on molecular runs). PyTorch only: the same pinning REGRESSES CuPy (+7.5% at
240 AO, +5.2% at 580 AO) because CuPy already stages through a pinned pool.

**1. Reuse a pinned host buffer.** `t.cpu().numpy()` targets pageable memory, so
`cudaMemcpyAsync` is staged through a bounce buffer by the CPU, synchronously.
D2H goes from 3.22 to 192.7 GB/s (nsys, 65.8 GB in 525 ops).

| ladder | baseline | pinned | | CCSD | baseline | pinned | |
|---|---:|---:|---:|---|---:|---:|---:|
| N=800 | 286.52 | 233.06 | -18.6% | 240 AO | 105.87 | 66.35 | -37.3% |
| N=1100 | 1473.39 | 1030.72 | -30.0% | 580 AO | 915.90 | 810.79 | -11.5% |
| N=1200 | 1948.04 | 1583.43 | -18.7% | 920 AO | 3974.28 | 3740.86 | -5.9% |
| N=1300 | 2909.81 | 2236.56 | -23.1% | | | | |

N=1200 and 1300 go from **0.96x and 0.92x of their H100 C-split to 1.18x and
1.20x**; their GH200 could not run those cells at all. Peak RSS +0.9 GiB.

**The molecular benefit decays with basis size** because the fix acts on
`td_GPU_helper` (`GPU: Generic`), not on C-split: at 920 AO Generic falls
1016.3 -> 776.4 s while C-split does not move (1809.4 -> 1816.6). C-split's
share rises with basis size, so the unreachable share rises with it.

**2. Make `unravel` cheap.** `assign_triu` builds `np.triu_indices(nacto*nactv)`
per call -- 9.02 GiB of int64 indices plus a 605 M-element scatter at 920 AO --
where the triangle is row-contiguous; and `iadd_transpose` materialises two full
temporaries when `(2,3,0,1)` on an `(o,v,o,v)` array **is** matrix transpose on
the `(ov,ov)` view, i.e. `M += M.T`, blockable in place. Bitwise identical;
block 1024 (4096 is 3x worse -- cache).

| AO | unravel | | CCSD (both cells pinned) | |
|---:|---:|---:|---:|---:|
| 580 | 140.2 -> 22.5 s | 6.2x | 800.99 -> 637.62 s | -20.4% |

**Together at 580 AO: 915.90 -> 637.62 s, -30.4%**, i.e. 159.40 s per iteration
against their GH200 PyTorch 5.7 m = **2.15x**, up from 1.58x.

**The biggest remaining lever is upstream, not a patch.** `crosslib_batching`
consumes transfers two ways: `dest[...] += move_tensor_to_cpu(part)` at
437/489/998/1062/1487 and slice-assign at 541/1117 already know the destination,
so the DMA could write it directly; at 351/582/925/1158 the array is returned and
escapes, which is why our patch must copy. Measured: **10.8 GB/s staged vs
193 GB/s into a registered destination**, an 18x gap that nsys cannot see because
the copy is host-side.

## PyBEST bugs and traps found (worth reporting upstream)

1. **`log.set_level()` does not exist.** `level` is a property setter, so it is
   `log.level = log.high`. Every doc example implying otherwise is wrong.
2. **SCF crashes at `log.do_high`.** `scf_diis.py:254` calls
   `self._history.log(coeffs)`, and `DIISHistory.log` does
   `min(state.energy for state in ...)` guarding only `EmptyData`; when the
   energies are `None` it escapes as `TypeError` (`scf_diis.py:480`). The
   `None` case is explicitly anticipated three lines later, so it is a missing
   `except`. This is the verbosity required to detect silent CPU fallback, so
   the diagnostic you need is the one that crashes. Workaround: raise verbosity
   only after SCF.
3. **`filemanager.temp_dir` defaults to a path relative to the CWD.** On a
   cluster that puts multi-GB HDF5 checkpoint spills wherever the job happened
   to `cd` -- for us, Lustre. At 920 AO that produced
   `FileNotFoundError: Cannot find checkpoint file ...` on read-back. Setting
   it to node-local NVMe fixed it (job 147598 vs 147348). PyBEST would benefit
   from honouring an environment variable here.
4. **`splitting_assistance` returns its result; it does not write into the
   array passed as the last operand.** `base.py` uses
   `arr[slice_] += factor * splitting_assistance(...)`. Passing an output array
   and reading it back silently yields zeros.
5. **`assign_triu` builds `np.triu_indices` on every call** -- 9.02 GiB of
   int64 index arrays at 920 AO, for a triangle that is row-contiguous. 7.1x
   available for free (`dense_four_index.py:490`).
6. **`iadd_transpose` materialises two full temporaries**
   (`dense_four_index.py:828`); for `(o,v,o,v)` with `(2,3,0,1)` it is
   `M += M.T` on the `(ov,ov)` view and blocks in place. 5.0x available.
7. `os.cpu_count()` reports the whole node (144), not the Slurm allocation.
8. Timer output is labelled "CPU time usage" and stored in a field named
   `.cpu`, but `Timer` uses `time.perf_counter()` -- it is **wall time**.

## Corrections to earlier notes in this file

- **`n.c.` means two different things in the paper.** Table 4: insufficient
  memory. Table 5: *not converged* -- "we encountered internal errors with
  PyTorch". Do not treat Table 5 gaps as a memory ceiling to beat.
- Table 4's `n.c.` is **not purely CPU-side**: the footnote says "insufficient
  memory on the CPU side **or OutOfMemoryError for X-split**".
- Their X-split was deliberately left tuned for a 32 GB V100S "for reasons of
  reproducibility". Beating X-split is not a hardware result.
- **`geminals` (pCCD) has ZERO GPU-eligible contractions**, and neither do `pt`
  or `sapt`. pCCD's amplitudes are 2-index pairs, so it never emits the 4-index
  ladder. The modules that do are `cc` (31 call sites), `ip_eom` (73),
  `ea_eom` (21), `ee_eom` (15).
- **Only a `CholeskyFourIndex` operand can reach the GPU.**
  `CholeskyFourIndex.einsum_index('abcd')` returns `'xac,xbd'`, the dense one
  returns `'abcd'`, and every entry of `gpu_contraction_optimized` begins
  `xac,xbd,`. `rcc_cache.py` keeps "one whole CholeskyFourIndex to rule them
  all" and hands out views, so `from_cache("gvvvv")` stays Cholesky.
- **IP-EOM's GPU calls are all in one-time `set_hamiltonian_*` setup**, none in
  the Davidson sigma-vector routine -- so despite having the most call sites it
  should not be expected to speed up. `cc` has 14 in `cc_residual_vector`, the
  per-iteration hot path, and EA-EOM has some in `build_subspace_hamiltonian`.
- The paper's Table 5 includes **L0 / cc-pVTZ at 1004 AO** (GH200 CuPy 52.8 m),
  and a `PyBEST/CD/CuPy/GPU-only` variant that is *slower* than hybrid
  (59.6 m vs 52.8 m). Earlier notes here omitted both.
- Their H100 Table 5 rows mostly used **1 CPU core**; one 32-core row shows
  139.7 m -> 63.7 m for the same case. Only the GH200 rows (72 cores) are
  cleanly comparable.

## Working in this folder (MCP servers + skills)

**Already installed here** — `.mcp.json` and `.claude/skills/` are in place, so
opening this directory in Claude Code is all that is needed. The skills register
immediately; **the MCP servers only load at session start, so restart Claude Code
in this directory once.**

What is installed:
- `.mcp.json` — two stdio servers, `hpc` and `hpc-docs`, launched via
  `uv tool run --from git+https://github.com/william-dawson/hpc-agent-core.git@unified-hub`.
  No venv to maintain; it tracks the `unified-hub` branch.
- `.claude/skills/` — 8 skills: `hpc-facilities` plus the 7 `rikyu-*` packs
  (`configuring`, `demo`, `reference`, `remote-command`, `reproducing`,
  `submitting-jobs`, `monitoring-jobs`). Only the `rikyu` facility pack is
  installed, per the upstream instruction not to install all packs by default.

To reinstall from scratch (e.g. in another directory):
```bash
git clone --branch unified-hub --depth 1 \
    https://github.com/william-dawson/hpc-agent-core.git /tmp/hac
mkdir -p .claude/skills
cp /tmp/hac/plugins/hpc/.mcp.json .mcp.json
cp -R /tmp/hac/plugins/hpc/skills/hpc-facilities .claude/skills/
cp -R /tmp/hac/plugins/hpc-rikyu/skills/rikyu-* .claude/skills/
```
Facility settings live in `~/.hpc-agent/rikyu.json` (outside this folder, so
unaffected by any directory move):
```json
{"ssh": {"host": "rikyu"}}
```
Verify with `uv tool run --from git+...@unified-hub hpc-doctor rikyu`. Everything
should read `✓` except the embedding endpoint, which is non-blocking (docs search
falls back to BM25 keyword matching; results are then tagged `[search_method: bm25]`).

### Why this folder is not `~/Documents/AI4S/pbest`
macOS TCC revoked access to `~/Documents` mid-session. That blocked the files
**and** killed the `hpc` MCP tools, because the server process runs with its
working directory inside the project (it writes `remotemanager.log` there) and
died on its own cwd. If you restore Documents access, consolidate to one location
rather than letting the two copies drift.

## ⚠ `ssh rikyu 'cmd'` gives a NON-LOGIN shell — this has bitten us twice

Two failures that looked like cluster outages were both this:

1. **`apptainer: command not found`** — it lives in `/shared/software/apptainer/bin`,
   which is only on `PATH` in a login shell. Use the absolute path.
2. **`sinfo`/`squeue`/`sacct` failing with
   `resolve_ctls_from_dns_srv: res_nsearch error: Unknown host`** — RIKYU runs
   **configless Slurm**, and the login shell sets
   `SLURM_CONF_SERVER=sctl1:6817,sctl2:6817`. Without it Slurm falls back to a DNS
   SRV lookup that fails. **Slurm is fine; the invocation was wrong.**

**Always wrap direct-ssh cluster commands in a login shell:**
```bash
ssh rikyu 'bash -lc "squeue -u $USER"'
```
This matters doubly because a failed `squeue` returns **empty**, which is
indistinguishable from "job finished" — it produced a false "JOB ENDED" in an
early monitor. Never suppress stderr on Slurm commands, and confirm job death
via `sacct` state, not an empty `squeue`.

The `hpc` MCP tools use a login shell internally, so they were never affected —
this is only a hazard for direct `ssh`.

## Prior art — this benchmark exists, one GPU generation back

**The PyBEST docs contain no performance data.** Across 81 `.rst` files: zero hits
for "speedup"/"timing"/"wall time"/"flop"; all three "benchmark" hits are
bibliography entries. The data is in two papers the GPU page cites:

- **`[kriebel2024]`** JCTC **20**, 1130–1142 (2024) — original CPU-vs-GPU work.
- **`[dobrowolska2026]`** *"Efficient Coupled-Cluster Python Frameworks for
  Next-Generation GPUs: A Comparative Study of CuPy and PyTorch on the Hopper and
  Grace Hopper Architecture"*, **JCTC 2026, 22, 6533–6546** (arXiv:2603.20912).
  Dobrowolska, Świerczyński, Tecmer, Sujkowski, Ahmadkhani, Mazur, Noga,
  **Hammond** (NVIDIA), Boguslawski. PDF: `acs.jctc.6c00558.pdf`;
  extracted text: `dobrowolska2026-jctc.txt`.

They used **PyBEST v2.2.0.dev0** — essentially our release — on PLGrid machines.
(That also explains the `plgkbogusla/plgrid` tarball ownership that broke our
build; see the TAR_OPTIONS trap below.)

### They have NO Blackwell numbers
4 "Blackwell" mentions, **0** for B200/GB200, vs 46 GH200 and 37 H100. Blackwell
appears once in the Introduction as the step they did not take:

> "More computational advantage is to be offered within NVIDIA's Blackwell
> architecture, particularly when exploiting **FP64 emulation**…"

cited to Uchino/**Ozaki**/Imamura (INT8 emulation; Ozaki-II with FP8),
Brower/**Hammond**/Legeza (Blackwell emulated-FP64 DMRG, JCTC 2026), and
**Dawson, Ozaki, Domke, Nakajima, JCTC 2024, 20, 10826** — *this user's own
paper*. The gap they name is exactly where our work sits, and **FP64 emulation is
a first-class axis**, not an afterthought.

### Their setup (match it)
Cholesky threshold **1e-5** (PyBEST's default is 1e-4); frozen core (1s on
C/N/O); **nchol ≈ 5×(nocc+nvirt)**. Systems: (H2O)10 at 240/580 AO, (mU)2·H2O at
468 AO (6-31+G**), L0 dye at 444/742/1004 AO. They also used *synthetic
dimension-matched tensors*, which legitimises matching dimensions rather than
exact molecules.

**Metric:** mean time of **ONE CC iteration averaged over 4–5 steps** (vector
function + amplitude update + energy). GPU times include CPU-side batching prep,
transfer, and the algebra. So cap `maxiter` — **do not converge**.

### Their numbers (Table 5, molecular CCSD)
| system / basis | Grace CPU 72c | GH200 CuPy | GH200 PyTorch |
|---|---|---|---|
| (H2O)10 / cc-pVDZ (240) | **52 s** | 23.9 s | 25.4 s |
| (H2O)10 / cc-pVTZ (580) | — | 5.5 m | 5.7 m |
| (mU)2·H2O / 6-31+G** (468) | **18.2 m** | 6.1 m | 6.7 m |
| L0 / cc-pVDZ (444) | **8.5 m** | 3.1 m | 3.3 m |
| L0 / aug-cc-pVDZ (742) | — | 19.7 m | 17.1 m |

**The pure-CPU `PyBEST/CD/Grace` row is our single best reproduction target** —
GB200 has the same Grace CPU as GH200, so a mismatch means our setup is wrong,
not our hardware different.

**Calibrate expectations: GPU beats 72 Grace cores by only ~2–3×** (52→23.9 s,
18.2→6.1 m, 8.5→3.1 m). The "3–16×" in the abstract is against their *own older
GPU-CPU hybrid*, not a CPU. For scale, TeraChem/CD on a V100 does (H2O)10 in
10 s — faster than every PyBEST entry.

### Their Table 4 (synthetic ladder contraction `abcd,ecfd->efab`), seconds
nocc=100, nvec=5N rows (the main series):

| N | lib | GH200-X | GH200-C | H100-X | H100-C |
|---|---|---|---|---|---|
| 800 | PyTorch | 574.7 | 355.9 | 789.4 | 330.5 |
| 800 | CuPy | 987.7 | 306.6 | 933.4 | 461.5 |
| 900 | PyTorch | 1265.7 | 587.1 | 1753.3 | 496.2 |
| 900 | CuPy | 2450.2 | 495.0 | 1853.0 | 809.7 |
| 1000 | PyTorch | 2673.7 | 805.8 | n.c. | 827.1 |
| 1000 | CuPy | 5241.7 | 829.1 | n.c. | 1338.7 |
| 1100 | PyTorch | 5177.4 | 1401.7 | 7907.7 | 1223.4 |
| 1200 | PyTorch | n.c. | n.c. | 14821.7 | 1864.7 |
| 1300 | PyTorch | n.c. | n.c. | 25244.9 | 2679.1 |

They also ran nocc ∈ {50,150,200,225,250} and nvec ∈ {5N,10N}.

**C-split is NOT universally better.** At nocc=200/nvec=10N, GH200 PyTorch
X-split (834.7 s) *beats* C-split (1573.6 s). The winner is regime-dependent.

**"n.c." in Table 4** = "insufficient memory on the CPU side **or
OutOfMemoryError for X-split**" (GH200 480 GB; H100 node 1006 GB) — so for
X-split it can be VRAM. In **Table 5 it means something else entirely: *not
converged***. Rikyu grants 400 GB per GPU requested, so `--gpus=2` (800 GB) or
`--gpus=4` (1600 GB) can compute Table 4 cells neither of their machines could.

## Target hardware (from `get_facility` + measured)

**Node: NVIDIA GB200 NVL4** — 2× Grace CPU + 4× B200.

| | |
|---|---|
| GPU HBM3e | **184.0 GiB per GPU** (measured in-container; `get_facility` says 173.2 — trust the measurement) |
| GPU mem bandwidth | ~7.9 TB/s per GPU |
| CPU | Neoverse-V2, 2 sockets × 72 cores = 144/node, LPDDR5X 960 GiB, 768 GB/s |
| **NVLink-C2C** | **450 GB/s bidirectional, cache-coherent** |
| Node-local NVMe | 6.7 TB free at `/tmp`, ~7.2 GB/s |

System: 400 nodes, 1600 GPUs, **64.160 PFLOPS FP64**, 15.539 EFLOPS FP8 →
**~40 TFLOP/s FP64 per B200**, roughly a 1:242 FP64:FP8 ratio. Against ~1–2
TFLOP/s for 32 Grace cores the GPU still leads ~an order of magnitude on FP64 and
~10× on bandwidth — so the open question is **whether PyBEST's offload realizes
that headroom**, not whether Blackwell can do FP64.

vs GH200: **184 GiB HBM vs ~96 GiB**, and up to 1600 GB host RAM vs 480 GB.

## Verified live on RIKYU (2026-09-28)

| | |
|---|---|
| login node OS | **Ubuntu 24.04.5 LTS**, `aarch64`, **glibc 2.39** |
| g++ | 13.3.0 |
| system Python | 3.12.3 |
| `uv` | preinstalled at `~/.local/bin/uv` |
| NVIDIA driver | **580.178.04** (CUDA 13 era → `cupy-cuda13x`) |
| apptainer | **1.4.5-3.el8** at `/shared/software/apptainer/bin/` |
| GPUs on login node | 4× GB200 visible |

Host packages present: `libboost-dev 1.83`, `libopenblas-dev 0.3.26` (pkg-config
works; `cblas.h` at `/usr/include/aarch64-linux-gnu/`), `libgmp10`.
**Absent: eigen3** (supplied by the container's apt).

**Storage — the bundled guide is wrong; trust these:**
- Home limit is **50 GB, not 5 GB** — but 35 GB used, ~15 GB free. Not for builds.
- Group quotas vary: `rkp00012` 4.2/10 T, `rkp00015` 13.7/27.3 T, `rkp00040` 5.5 G/1 T.
- **We use `/data1/rkp00012`.**

## Slurm gotchas (learned by submitting)

- **`--account=rkp00012` is MANDATORY.** `get_facility` says
  `"account_required": false` and the guide says omit it — **both wrong** for a
  user in multiple projects. sbatch rejects the job otherwise.
- **1 GPU yields 32 usable cores, not the documented 36** (matches GPU0's
  affinity mask `0-15,28-43`). 2 GPUs → 64, 4 → 128.
- `free` inside a job reports whole-node memory, not the Slurm limit.
- Likewise **`os.cpu_count()` reports 144**, the whole node, not the allocation
  (`cpus_visible: 144` in a `--gpus=2` run that really had 64). `nproc` is
  correct. Use `$SLURM_CPUS_ON_NODE` or `nproc` when recording provenance.
- **Slurm's control plane can be unreachable from the login node**
  (`resolve_ctls_from_dns_srv … DNS SRV lookup failed`). Then `squeue`/`sacct`
  return **empty regardless of job state** — do NOT read that as "job finished".
  Never suppress their stderr. Check liveness by ssh-ing to the compute node.

## Topology for NUMA binding (node c180, `--gpus=1`)
- **34 NUMA nodes**: node0 = CPUs 0–71, node1 = 72–143; the other 32 are **GPU HBM
  exposed as NUMA nodes** (NVLink-C2C coherency made visible).
- GPU0: CPU affinity **`0-15,28-43`**, NUMA affinity 0, **GPU NUMA ID 2**.
- Bind to GPU0's affinity mask; `numactl` can target GPU memory directly, which
  makes the managed-memory axis concretely testable.

## Container

Built by `container/build-container.sh` (staged, resumable) then patched by
`container/fix-container.sh` and `container/add-pkgs.sh`. Final artifact:
**`/data1/rkp00012/rku00036/pybest/pybest-rikyu-v3.sif` (4.8 GB)**.

Verified in-container:
```
PYBEST 2.2.0 | numpy 2.5.3 | scipy 1.18.1 | opt_einsum 3.4.0
CHOLESKY_ENABLED True        GPU_PATTERNS 18
CUPY 14.2.0 dgemm_ok  VRAM 184.0 GiB total
TORCH 2.14.0+cu130  cuda_available True  devices 4  (NVIDIA GB200)
```

**Invocation (both flags are required):**
```bash
A=/shared/software/apptainer/bin/apptainer      # ssh 'cmd' has no apptainer on PATH
$A exec --nv -B /data1 \                        # /data1 is NOT bound by default
   --env PYBEST_CUPY_AVAIL=1,PYBEST_C_SPLITTING=1 \
   /data1/rkp00012/rku00036/pybest/pybest-rikyu-v3.sif python script.py
```

### Build gotchas (each cost a failed job)
- **apt must be routed off `/tmp`**: `TMPDIR=/var/tmp` + `-o Dir::Temp=/var/tmp`.
  Slurm gives each job a **private `/tmp` at mode 0700** (`drwx------`), unlike the
  login node's `1777`; apptainer bind-mounts it into `%post`, apt drops to the
  `_apt` user, which under `--fakeroot` maps to an unmapped uid → every repo reads
  "not signed", `apt-get update` exits 100. **A login-node container test does NOT
  validate a compute-node build.**
- **`apptainer build` has no `--no-mount`** (only `exec`/`run`/`shell` do; they
  also take `--env`). Passing it to `build` just prints usage.
- **`export TAR_OPTIONS=--no-same-owner` is REQUIRED.** The libint/libchol
  tarballs record owner `plgkbogusla/plgrid`; under `--fakeroot` only one uid is
  mapped, so GNU tar as root fails the chown and exits 2 — and
  `depends/Makefile:125` sends tar's stderr to `/dev/null`, so it dies with a bare
  `Error 2` *after* a clean 567 MB download. Also `sed` that `2>&1` out.
- **`cmake` must be in the image.** The pip cmake only lands in the venv, which
  doesn't exist when libint builds.
- **`cupy-cuda13x[ctk]`**, not bare `cupy-cuda13x` — CuPy needs CUDA toolkit
  headers at runtime for kernel JIT.
- **A staged sandbox inherits the *base* def's `%environment`** — ours was empty,
  so the .sif had no `LD_LIBRARY_PATH` and PyBEST failed with
  `ImportError: libchol.so`. Fixed with `ld.so.conf` + `ldconfig` **and** a baked
  `/.singularity.d/env/91-pybest.sh`.
- **Bind-mounting into a `--writable` sandbox fails** if the destination doesn't
  exist; just `cp` into the sandbox directory instead.
- Build in node-local `/tmp` (6.7 TB NVMe), not Lustre — libint makes thousands of
  small files. Copy only the finished `.sif` to `/data1`.
- Slim before freezing: `/opt/src` holds the unpacked libint source + build tree.
- **Cache the libint install tree** (`cache/libint-install.tar`, 435 MB) so no
  later failure forces a rebuild. **libint took 35 min** on 144 cores; one
  `cc1plus` held 3.5 GB for 25 of them on a single unity TU.
- Build with **`--gpus=4`**: no GPU is used, but GPU count is how you buy cores.

## PyBEST facts that drive everything

### Dependency tiers
- **Tier 1 (compiled):** libint2 2.11.2 (**567 MB** tarball, the long pole);
  libchol 0.1.12 (506 KB, **mandatory** — no libchol → no Cholesky ERI → no GPU
  path); `pybest.core` (nanobind).
- **Tier 2:** Eigen3 (header-only), cmake, OpenMP via libgomp. Boost/GMP are
  libint's build deps only — GMP only for its *generator*, not the pre-generated
  tarball.
- **Tier 3 (all aarch64 wheels exist):** numpy, scipy, h5py, cupy-cuda12x/13x.
  Python floor is **3.10** (`cp310`–`cp314`), not the 3.13 the Linux docs imply.
  **Drop PySide6** — only `daisy/picker.py` uses it, behind `try/except`, and
  `pybest/__init__` never imports `daisy`.

### Source traps
- **`PYBEST_USE_MKL` / `PYBEST_USE_OPENBLAS` are no-ops.** `USE_MKL`,
  `USE_OPENBLAS`, `USE_OPENBLAS_PC` are declared in `core/CMakeLists.txt:18-20`
  and read nowhere; no `cblas`/`blas` include exists in any `core/*.cpp`. The
  install docs' MKL ceremony is vestigial. BLAS matters only via **numpy** and
  **libchol**.
- **`depends/Makefile` hardcodes `OPENBLAS_TARGET = x86_64`** on Linux (plain `=`);
  only a command-line override works. Moot if using apt's OpenBLAS.
- **`CXX? = g++`** (stray space) defines a variable named `CXX?`, so `CXX` is never
  set on Linux and `-DCMAKE_CXX_COMPILER=` goes out empty. Set `CXX` explicitly.
- Docs' only quantitative note: `DenseLinalgFactory` needs ~`3N^4` (→`4N^4` with
  ERI). At N=1000 that's ~32 TB, so **Dense is impossible; Cholesky is mandatory**.

## The GPU model (the crux)

Offload is **not** per-method. It is one choke point: `NIndexObject.contract()` in
`src/pybest/linalg/base.py`, dispatching on `select`:
- `select="td"` → **forces CPU**.
- `select="cupy"`/`"pytorch"` → explicit GPU via `splitting_assistance`.
- `select=None` → **automatic**: if the subscript is one of the 18 patterns in
  `_gpu_support.py:gpu_contraction_optimized` *and* a backend is available → GPU.
- Dense operands with `ndim <= 2` are deliberately skipped
  (*"CuPy is not faster for dense matrices … (yet)"*).

All 18 patterns are `xac,xbd,<…>` — CC ladder terms with two Cholesky vectors.
**Modules that actually reach the GPU** (counted by matching call sites against
the pattern list): `cc` 31, `ip_eom` 73, `ea_eom` 21, `ee_eom` 15.
**`geminals` (pCCD), `pt` and `sapt` have ZERO** — pCCD's amplitudes are 2-index
pairs, so it never emits the 4-index ladder. Benchmarking bare pCCD measures a
CPU path. Note also that `ip_eom`'s calls are all in one-time
`set_hamiltonian_*` setup rather than the Davidson sigma-vector routine, so its
high count does not translate into speedup; `cc` has 14 in `cc_residual_vector`,
the per-iteration hot path.

Runtime control is **environment only**, read at import:
- `PYBEST_CUPY_AVAIL=1` / `PYBEST_PYTORCH_AVAIL=1` (CuPy wins if both)
- `PYBEST_C_SPLITTING=1` xor `PYBEST_X_SPLITTING=1` (defaults to C if neither/both)

Direct entry point for microbenchmarks:
`splitting_assistance(subs, op1, op2, op3, out)` — first two are Cholesky
tensors; `parts=` kwarg controls granularity. **The last operand is NOT filled
in: the function RETURNS the result** (`base.py` does
`arr[slice_] += factor * splitting_assistance(...)`). Reading the array you
passed gives zeros.

Separate knob: **`indextrans=`** on the method call selects the 4-index AO→MO
transformation backend (`"cupy"`/`"tensordot"`/`"einsum"`) and has **its own
silent fallback**.

### ⚠ Silent CPU fallback — the main methodological hazard
Every GPU path degrades to CPU **without failing**: `select=None` wraps GPU calls
in a bare `except Exception`, and `MemoryError` falls back to `td_helper` with a
warning emitted **only at `log.do_high`**. A "GPU run" can produce perfectly
correct numbers having never touched the GPU.

**Never infer GPU use from the env var being set.** Corroborate every GPU point
with (a) `log.level = log.high` output — NOT `log.set_level()`, which does not
exist, and only *after* SCF, which crashes at that verbosity — and
(b) `nvidia-smi` sampling or profiler counters.

### Measure per-section, not wall clock
PyBEST registers `log.print_footer` via `atexit`, which calls `timer.report()` and
prints a **per-section timing table**. Harvest it: libint integral evaluation and
the libchol Cholesky decomposition are CPU-only regardless of GPU settings and
dominate at small nbasis, so wall clock lets Amdahl's law bury the offload benefit.

## Benchmark design

**Phase 1 — reproduce.** Validate the setup against published numbers before
claiming anything new.
1. `bench/repro_table4.py` — their synthetic ladder contraction at exact published
   dimensions. No SCF, no geometry, no CCSD; isolates the one thing being
   measured. **Start at N=800, nocc=100, nvec=5N** (GH200 refs: PyTorch C-split
   355.9 s, CuPy C-split 306.6 s).
2. `PyBEST/CD/Grace` CPU-only baseline — (H2O)10/cc-pVDZ, 72 cores, **52 s**.
   Same CPU family, so this is the cleanest possible validation.
3. Molecular CCSD (Table 5 rows) via `bench/gen_systems.py`.

**Phase 2 — extend.** Fill their `n.c.` cells (we have more host RAM), push past
nbasis 1004, and test **FP64 emulation** — the axis they explicitly flagged.

**Status of Phase 2:** emulation is done and the answer is that it buys little
(~1.15% at best inside PyBEST, see Results) because the GEMM is only ~20% of the
contraction. 920 AO is past their largest *water* case but still under their
1004 AO L0 point. N=1200/1300 were dropped on cost, not blocked.

### Job sizing
Host-array budget for the ladder contraction (nocc=100, nvec=5N):
| N | arrays | GPU request |
|---|---|---|
| 800 / 900 / 1000 | 102 / 138 / 181 GiB | `--gpus=1` |
| 1300 | 354 GiB | `--gpus=2` |

But **their GH200 runs used 72 CPU cores and `--gpus=1` gives us only 32**, and
their metric includes CPU-side batching prep. **Run the reproduction at
`--gpus=2`** (64 cores, 800 GB) for comparability, or `--gpus=3` (96 cores) and
pin to 72 threads for an exact core match.

### Axes
1. CPU vs GPU · 2. CuPy vs PyTorch · 3. C- vs X-splitting · 4. problem size ·
5. thread count · 6. managed vs explicit memory · 7. **FP64 emulation**.

### Systems (`bench/gen_systems.py`)
PyBEST ships 62 geometries but the largest is ~54 atoms — unit-test scale. We
generate our own: **(H2O)n water clusters** (reproduce their exact 240/580 AO at
n=10) and **linear all-anti n-alkanes** (closed-shell, large HOMO–LUMO gap, so RHF
converges reliably; carbon clusters were rejected as multireference). Geometries
are idealized, not optimized — correct for cost benchmarking, where all that
matters is that systems are reasonable and identical across compared configs.
Validated: exact 1.526 Å C–C, 1.090 Å C–H, 0.9572 Å O–H, no clashes.

Grow the **basis**, not the molecule: cc-pVQZ reaches nbasis ~1900 at 16 carbons
where cc-pVDZ needs ~80, and it inflates **nvirt**, which dominates CC cost.

Method must match regime: **CCSD is primary** (it is what they measured, and the
per-iteration metric keeps it tractable); **pCCD** is a cheap companion that
reaches deeper into the batching regime.

## Repository layout

```
container/   apptainer definition + staged build scripts + verify2.py
bench/       repro_table4.py, bench_fp64.py, gen_systems.py
jobs/        Slurm submission scripts
systems/     .xyz geometries (the drivers beside them are generated and
             gitignored -- regenerate with bench/gen_systems.py)
results/     table4/ (synthetic) and ccsd/ (molecular)
```

**The copy on RIKYU is flat**, not mirrored: everything sits directly in
`/data1/rkp00012/rku00036/pybest/`, which is what the `jobs/*.sh` scripts
reference via `P=`. Mirror the directories there before relying on the paths
matching, and do not replace a script while a job is executing it -- bash
reads scripts incrementally.

## Operating rules
- Nothing heavy on the login node. Builds and benchmarks go through Slurm.
- **Cost is real**: 300 yen/GPU-hour, billed per GPU requested, idle or not.
- RIKYU is in Early Access Phase 2 (through end of Sept 2026) — re-verify policy
  claims rather than trusting the bundled guide, which has been wrong on storage
  quotas, account requirements, and core counts.
- The `hpc` MCP tools break if `~/Documents` is inaccessible (the server's cwd
  lives there). Direct `ssh rikyu` works as a fallback.
