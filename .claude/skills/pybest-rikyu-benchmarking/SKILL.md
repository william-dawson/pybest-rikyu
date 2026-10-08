---
name: pybest-rikyu-benchmarking
description: Use when running or planning a PyBEST benchmark on RIKYU (GB200) - sizing a job, invoking the container, choosing GPU count, estimating cost and wall time, writing a measurement that is trustworthy, or debugging a failed benchmark job. Covers the container, Slurm specifics, job sizing arithmetic, measurement methodology and the traps that have each cost a run.
user-invocable: true
---

# Benchmarking PyBEST on RIKYU

## 1. Container and invocation

Artifact: `/data1/rkp00012/rku00036/pybest/pybest-rikyu-v3.sif` (4.8 GB), built
from `container/pybest-rikyu.def` on `ubuntu:24.04`. **No RIKYU modules are used
by any benchmark** - CUDA comes from the image's pip wheels
(`cupy-cuda13x[ctk]`, `torch` resolved to 2.14.0+cu130) plus the host driver
580.178.04 injected by `--nv`. Checking RIKYU's `nvhpc` modules tells you nothing
about these runs.

```bash
A=/shared/software/apptainer/bin/apptainer   # ssh 'cmd' has no apptainer on PATH
$A exec --nv -B /data1 \                     # /data1 is NOT bound by default
   --env PYBEST_CUPY_AVAIL=1,PYBEST_C_SPLITTING=1,PYBEST_TEMP=/tmp/pb-$SLURM_JOB_ID \
   $SIF python script.py
```

Verified in-container: PyBEST 2.2.0, numpy 2.5.3, Cholesky enabled, 18 GPU
patterns, CuPy 14.2.0, PyTorch 2.14.0+cu130, 184.0 GiB VRAM per GPU.

**`PYBEST_TEMP` is not optional for large runs.** `filemanager.temp_dir` defaults
to `pybest-temp` relative to the CWD, so a job run from `/data1` spills multi-GB
HDF5 checkpoints onto Lustre. At 920 AO that produced
`FileNotFoundError: Cannot find checkpoint file ...` on read-back; pointing it at
node-local NVMe fixed it. The driver template honours the variable:

```python
from pybest import filemanager
if os.environ.get("PYBEST_TEMP"):
    filemanager.temp_dir = os.environ["PYBEST_TEMP"]
```

Node-local `/tmp` is 3.4 TB of NVMe, auto-deleted at job end. PyBEST removes its
own temp dir at exit, so `du` after the run reports nothing.

## 2. Slurm specifics

- **`--account=rkp00012` is MANDATORY.** `get_facility` reports
  `account_required: false`; that is wrong for a user in several projects.
- **GPU count is how you buy CPU cores and memory.** 1 GPU -> 32 usable cores
  (not the documented 36) and ~400 GB. 2 -> 64, 3 -> 96, 4 -> 128.
- **PyBEST is single-GPU.** GPU 1 has sat at 0% and ~9 MiB in every job. We
  request 2 GPUs only to buy cores and host memory, and pay 300 yen/GPU-hour for
  a card that does nothing. Factor that into any cost estimate.
- `free` reports whole-node memory, and `os.cpu_count()` reports 144 - neither
  respects the allocation. `nproc` and `$SLURM_CPUS_ON_NODE` are correct.
- Billing is per GPU **requested**, idle or not.

## 3. Sizing a job

Operands for the synthetic ladder (`nocc`, `nvec_factor` as given):

```python
v, x = N - nocc, nvec_factor * N
bytes = 8 * ((2*x + 2*nocc**2) * v*v)      # xac + xbd + ecfd + out
```

`bench/repro_table4.py --dry-run` prints this exactly. Its internal refusal check
compares against `/proc/meminfo`, which shows the **whole node**, so it will not
protect you - size the GPU count yourself.

**Peak resident memory is not a fixed multiple of the operands** and the ratio
climbs with N: measured 1.39, 1.54, 1.60, 1.85, 1.79 at N=900/1000/1100/1200/1300
for CuPy (PyTorch runs lower, 1.41-1.71). Budget **1.9x** and verify.

| N | operands | peak @1.9x | GPU count |
|---:|---:|---:|---:|
| 1100 | 231 GiB | 439 | 2 |
| 1200 | 288 GiB | 548 | 2 |
| 1300 | 354 GiB | 673 | 2 |
| 1400 | 428 GiB | 813 | 3 |
| 1700 | 706 GiB | 1341 | 4 (the whole node) |

Time scales as roughly **N^5** for fixed nocc and nvec=5N (measured exponent
4.6-5.1 between adjacent points, N=800-1300). From the CuPy anchor 1051 s at
N=1100: multiply by `(N/1100)^5`. This is an empirical fit over a narrow range,
not a scaling law - it has no predictive standing outside N~800-1400, and it does
not account for the splitting-strategy change near N=1082.

For molecular CCSD, time per iteration scales as about `v^2.4` at small bases
rising toward `v^3.2` - three points only, so use it for ordering jobs, not for
quoting. The FLOP model in `pybest-ccsd-cost-model` is more principled but is
itself derived rather than measured; see the confidence note at its top.

## 4. Making a measurement you can trust

- **Match the published metric**: mean time of ONE CC iteration over 4-5 steps.
  **Cap `maxiter`; do not converge.** GPU times include CPU-side batching prep,
  transfer and the algebra.
- **Harvest the per-section timer**, not wall clock. It is printed at exit via
  `atexit -> log.print_footer`. Labelled "CPU time usage" but it is **wall time**
  (`Timer` uses `time.perf_counter()`). `GPU: C-split` vs `GPU: Generic` is the
  most informative split available without a profiler.
- **Corroborate GPU use per cell.** Sample
  `nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw`
  into a CSV during the run. Include `utilization.memory`: at N<=1200 it reads
  18-25% while `utilization.gpu` reads 100%, which is how you learn the kernel is
  not bandwidth-bound.
- **Constant operand fill is valid for FP64 timing and invalid for emulation.**
  `np.full(shape, 0.1)` is memset-speed and dense-GEMM FLOPs are data-independent
  - but cuBLAS ADP picks its mantissa count from the data's dynamic range, and
  constant data has none. Use `--random` for any precision work.
- **Reproducibility is good** when you follow the above: N=1100 reproduced to
  0.5-0.8% across different nodes four days apart, and CPU-side timer sections
  reproduce to under 1% between backends.
- **Variance is NOT good at small sizes.** At 240 AO the identical CPU-only SCF
  took 16.1 s and 5.6 s in back-to-back cells. Do not quote single runs below
  ~500 AO.

## 5. Traps that have each cost a run

| trap | symptom | fix |
|---|---|---|
| `log.set_level(log.high)` | `AttributeError` | `log.level = log.high` - it is a property |
| high verbosity during SCF | `TypeError: '<' not supported between NoneType` at `scf_diis.py:480` | raise verbosity only AFTER SCF |
| `pybest-temp` on Lustre | `FileNotFoundError: Cannot find checkpoint file` | set `PYBEST_TEMP` to node-local |
| reading the array passed to `splitting_assistance` | all zeros | it RETURNS the result |
| `ssh rikyu 'cmd'` | `apptainer: command not found`, Slurm DNS SRV errors | wrap in `bash -lc`; it is a non-login shell |
| empty `squeue` | looks like the job finished | configless Slurm may be unreachable; confirm via `sacct` state |
| grep-filtering a job's stdout | a traceback vanishes | write full output to a raw log, filter only for display |
| `echo "exit=$?"` after a pipe | reports grep, not python | capture `rc=$?` directly after the command |
| PyBEST's own log mid-run | nothing appears | it is block-buffered to a file; only your own `flush=True` prints are live |

## 6. Cost reality

300 yen per GPU-hour, billed per GPU requested. Rough costs actually incurred:

| experiment | GPU-h | yen |
|---|---:|---:|
| Table 4 N=800 (2 cells, 5 reps) | ~2 | ~600 |
| Table 4 N=900-1100 (6 cells) | ~13 | ~3,900 |
| Table 4 N=1100-1300 (6 cells, 2 reps) | ~11 | ~3,300 |
| CCSD 240 AO | ~0.1 | ~30 |
| CCSD 580 AO | ~1.5 | ~450 |
| CCSD 920 AO (x2 incl. one failure) | ~5 | ~1,500 |
| CCSD 1150 AO | ~13 | ~3,900 |
| precision probe | ~0.1 | ~30 |
| emulation sweep N=800 (8 cells) | ~3.5 | ~1,000 |

Reps matter: run-to-run spread on the ladder is 0.2-0.4%, so `--warmup 1
--reps 1` is adequate for a ratio and costs 40% less than the paper's 5 passes.

## Appendix: cuBLAS FP64/FP32 emulation on B200

Moved here from the README, which now carries only the main story. Keep it: the
reference paper names FP64 emulation as the Blackwell opportunity it did not
take, so it will come up, and the numbers below are what we can answer with.

cuBLAS 13.0u2+ emulates GEMM by two distinct mechanisms -- fixed-point
Ozaki-I/II for FP64, BF16x9 for FP32. Both are selected by environment variable,
so **PyBEST needs no changes**: CuPy and PyTorch call cuBLAS and cuBLAS decides
underneath. Knobs: `CUBLAS_EMULATE_DOUBLE_PRECISION`,
`CUBLAS_EMULATE_SINGLE_PRECISION`, `CUBLAS_EMULATION_STRATEGY`,
`CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT`. Measure with
`bench/precision_probe.py`, which runs each mode in a fresh subprocess because
cuBLAS reads the environment at handle creation.

Square DGEMM 8192^3, one B200, job 147643. Relative error is against a CPU
float64 reference, so the native row is the floor, not zero.

| mode | TFLOP/s | vs native | rel. error |
|---|---:|---:|---:|
| FP64 native | 38.4 | 1.00x | 1.6e-15 |
| FP64 emulated, automatic (ADP) | 50.7 | 1.32x | 1.6e-15 |
| **FP64 emulated, 55 mantissa bits** | **70.6** | **1.84x** | **4.9e-16** |
| FP64 emulated, 47 bits | 86.7 | 2.26x | 6.8e-14 |
| FP64 emulated, 39 bits | 111.4 | 2.90x | 1.6e-11 |
| FP64 emulated, 31 bits | 145.2 | 3.78x | 3.7e-09 |
| FP32 native | 69.4 | 1.81x | 8.1e-07 |
| FP32 emulated (BF16x9) | 191.2 | 4.98x | 2.1e-07 |

At 55 bits it is faster **and** more accurate than native FP64: fixed-point
accumulation avoids rounding that native DGEMM incurs while accumulating.
BF16x9 is likewise faster and more accurate than native FP32. The automatic
mode is conservative and under PyTorch is *slower* than native (0.91x) unless
`CUBLAS_EMULATION_STRATEGY=performant`; CuPy does not show this.

Inside PyBEST at N=800 (job 147646) most of the gain does not survive:

| mode | square DGEMM | CuPy in PyBEST | PyTorch in PyBEST |
|---|---:|---:|---:|
| automatic | 1.32x | 1.055x | 1.064x |
| 55 bits | 1.84x | 1.106x | 1.085x |
| 39 bits | 2.90x | 1.148x | 1.155x |

### Why this is not currently worth pursuing

**The measurement was taken in the wrong regime.** It used the paper's synthetic
dimensions, nocc=100, where `x/o^2 = 0.4` and integral reconstruction is only
29% of the ladder. A real large-basis case is the opposite: at 1150 AO
`x/o^2 = 5.25` and reconstruction is 84%. So these numbers say little about
where emulation would actually land.

**And emulation multiplies a quantity we have now shown is not the constraint.**
At 920 AO (job 162831) the GPU kernels total 1002.9 s of an 1882 s CCSD and run
at 82% of measured DGEMM peak, while `cudaFree` -- from PyBEST's
`clean_memory()` -> `torch.cuda.empty_cache()` -- costs 1049.4 s. Halving the
arithmetic would leave the allocator cost untouched.

Revisit only after the allocator cost is gone, and then measure at `o=40`, not
at the synthetic dimensions.
