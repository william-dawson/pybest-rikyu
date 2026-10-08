#!/usr/bin/env python3
"""Reproduce Table 4 of Dobrowolska et al., JCTC 2026, 22, 6533-6546, on GB200.

Their Table 4 times the CCSD particle-particle ladder contraction
`abcd,ecfd->efab` at synthetic dimensions with randomly generated arrays. In
Cholesky form that is `xac,xbd,ecfd->efab`, which appears verbatim in PyBEST's
`gpu_contraction_optimized` list, so we can drive it directly through
`splitting_assistance` -- no SCF, no geometry, no CCSD. That isolates exactly
the thing being measured, and is the cheapest possible first experiment.

Published reference (seconds, avg of 5), nocc=100, nvec=5N:
  N     lib      GH200-X   GH200-C    H100-X    H100-C
  800   PyTorch    574.7     355.9     789.4     330.5
  800   CuPy       987.7     306.6     933.4     461.5
  900   PyTorch   1265.7     587.1    1753.3     496.2
  900   CuPy      2450.2     495.0    1853.0     809.7
 1000   PyTorch   2673.7     805.8      n.c.     827.1
 1000   CuPy      5241.7     829.1      n.c.    1338.7
 1100   PyTorch   5177.4    1401.7    7907.7    1223.4
 1100   CuPy     11305.0    1282.7    8151.5    1860.9
 1200   PyTorch     n.c.      n.c.   14821.7    1864.7
 1300   PyTorch     n.c.      n.c.   25244.9    2679.1
 1300   CuPy        n.c.      n.c.   28292.9    4665.3
They also ran nocc in {50,150,200,225,250} and nvec in {5N,10N}.

NOTE: C-split is NOT universally better. At nocc=200/nvec=10N the GH200 PyTorch
X-split (834.7 s) BEATS C-split (1573.6 s). The winner is regime-dependent.

"n.c." = insufficient CPU-side memory (GH200 480 GB; H100 node 1006 GB), not
VRAM. Rikyu grants 400 GB per GPU requested, so --gpus=2 (800 GB) or --gpus=4
(1600 GB) can compute cells neither of their machines could.

Operand shapes for xac,xbd,ecfd->efab  (v=virtual, o=occupied, x=Cholesky):
    xac  (nchol, v, v)     xbd  (nchol, v, v)
    ecfd (o, v, o, v)      out  (o, o, v, v)

Run:
  A=/shared/software/apptainer/bin/apptainer
  $A exec --nv -B /data1 --env PYBEST_CUPY_AVAIL=1,PYBEST_C_SPLITTING=1 \
     pybest-rikyu-v3.sif python repro_table4.py --nbasis 800
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time


def shapes(n: int, nocc: int, nvec_factor: int = 5):
    nvirt, nchol = n - nocc, nvec_factor * n
    return {"xac": (nchol, nvirt, nvirt), "xbd": (nchol, nvirt, nvirt),
            "ecfd": (nocc, nvirt, nocc, nvirt), "out": (nocc, nocc, nvirt, nvirt)}


def gib(shape) -> float:
    n = 1
    for d in shape:
        n *= d
    return n * 8 / 2**30


def host_available_gib():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbasis", type=int, required=True)
    ap.add_argument("--nocc", type=int, default=100)
    ap.add_argument("--nvec-factor", type=int, default=5, choices=(5, 10),
                    help="Cholesky vectors as a multiple of N (paper: 5 and 10)")
    ap.add_argument("--reps", type=int, default=4, help="paper averages 4-5")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--parts", type=int, default=None,
                    help="pass through to the splitting routine")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the memory budget and exit")
    ap.add_argument("--random", action="store_true",
                    help="random normal operands instead of a constant fill. "
                         "REQUIRED for emulation runs: cuBLAS ADP picks its "
                         "mantissa count from the data's dynamic range, and "
                         "constant data has none")
    ap.add_argument("--seed", type=int, default=0,
                    help="operand seed; must match between a reference and the "
                         "runs compared against it")
    ap.add_argument("--ref-out", default=None,
                    help="save the result as .npy (put it on node-local NVMe)")
    ap.add_argument("--log-batching", action="store_true",
                    help="log free VRAM and the chosen batch counts; this is "
                         "how to compare what the two backends are told about "
                         "available memory")
    ap.add_argument("--ref-in", default=None,
                    help="compare against a saved reference, reporting the "
                         "relative Frobenius error")
    args = ap.parse_args()

    sh = shapes(args.nbasis, args.nocc, args.nvec_factor)
    total = sum(gib(s) for s in sh.values())
    avail = host_available_gib()
    print(f"N={args.nbasis} nocc={args.nocc} nvirt={args.nbasis-args.nocc} "
          f"nchol={args.nvec_factor*args.nbasis}")
    for k, v in sh.items():
        print(f"  {k:<5} {str(v):<26} {gib(v):8.1f} GiB")
    print(f"  {'TOTAL':<5} {'':<26} {total:8.1f} GiB"
          + (f"   (host available {avail:.1f} GiB)" if avail else ""))
    if args.dry_run:
        return 0
    if avail is not None and total > 0.85 * avail:
        print(f"REFUSING: needs {total:.1f} GiB, only {avail:.1f} GiB available. "
              f"Request more GPUs (Rikyu grants 400 GB per GPU).", file=sys.stderr)
        return 2

    # Import after the budget check -- importing pybest is not free.
    import numpy as np
    from pybest.linalg._gpu_support import (
        PYBEST_CUPY_AVAIL, PYBEST_PYTORCH_AVAIL, gpu_backend_select,
    )
    from pybest.linalg.crosslib_batching import splitting_assistance

    # Why does PyTorch degrade at large N? Batch counts come from
    # get_batch_sizes(chol_1, chol_2, mem_gpu) with mem_gpu = memory_usage(),
    # which is memGetInfo()[0] for CuPy and torch.cuda.mem_get_info()[0] for
    # PyTorch -- both DRIVER-level free VRAM, which excludes whatever each
    # library's caching allocator is holding. If PyTorch is sitting on more
    # cached device memory at that moment, PyBEST silently gives it more and
    # smaller batches on identical input. Log both so the decision is visible.
    if args.log_batching:
        import pybest.linalg.crosslib_batching as _cb

        _orig_gbs, _orig_mu = _cb.get_batch_sizes, _cb.memory_usage

        def _mu():
            v = _orig_mu()
            print(f"#batch free_vram_gib={v / 2**30:.2f}", flush=True)
            return v

        def _gbs(c1, c2, mem_gpu):
            n1, n2 = _orig_gbs(c1, c2, mem_gpu)
            print(f"#batch mem_gpu_gib={mem_gpu / 2**30:.2f} "
                  f"n_chol_1={n1} n_chol_2={n2} batches={n1 * n2}", flush=True)
            return n1, n2

        _cb.memory_usage, _cb.get_batch_sizes = _mu, _gbs

    c_split = bool(os.environ.get("PYBEST_C_SPLITTING", ""))
    x_split = bool(os.environ.get("PYBEST_X_SPLITTING", ""))
    print(f"\nbackend={gpu_backend_select!r} cupy={PYBEST_CUPY_AVAIL} "
          f"pytorch={PYBEST_PYTORCH_AVAIL} c_split={c_split} x_split={x_split}")
    if gpu_backend_select is None:
        # Not fatal -- a CPU number is a useful baseline -- but it must never be
        # mislabelled as a GPU result. PyBEST falls back silently.
        print("WARNING: no GPU backend active -- this is a CPU measurement.")

    # np.full is memset-speed; 0.1 avoids denormals. Dense-contraction FLOPs are
    # data-independent, so constant fill does not bias PLAIN FP64 timing.
    #
    # It DOES bias emulation. cuBLAS's ADP mode picks the mantissa bit count
    # from the data's dynamic range, and constant 0.1 has essentially none, so
    # ADP would choose the cheapest path and look artificially fast. Constant
    # data also has no cancellation, which would make every reduced-mantissa
    # mode look exact. --random is therefore REQUIRED for any emulation run.
    print(f"allocating ({'random normal' if args.random else 'constant'})...",
          flush=True)
    t = time.perf_counter()
    if args.random:
        rng = np.random.default_rng(args.seed)
        xac = rng.standard_normal(sh["xac"])
        xbd = rng.standard_normal(sh["xbd"])
        ecfd = rng.standard_normal(sh["ecfd"])
    else:
        xac = np.full(sh["xac"], 0.1)
        xbd = np.full(sh["xbd"], 0.1)
        ecfd = np.full(sh["ecfd"], 0.1)
    out = np.zeros(sh["out"])
    print(f"allocated in {time.perf_counter()-t:.1f}s", flush=True)

    subs = "xac,xbd,ecfd->efab"
    kw = {"parts": args.parts} if args.parts else {}

    # splitting_assistance RETURNS the contraction; it does not write into the
    # array passed as the last operand (base.py uses it as
    # `arr[slice_] += factor * splitting_assistance(...)`). `out` is there for
    # shape inference and stays zero, so the return value is the only result.
    times, result = [], None
    for i in range(args.warmup + args.reps):
        out[...] = 0.0
        t0 = time.perf_counter()
        result = splitting_assistance(subs, xac, xbd, ecfd, out, **kw)
        dt = time.perf_counter() - t0
        print(f"  {'warmup' if i < args.warmup else 'timed'} {i}: {dt:9.2f} s",
              flush=True)
        if i >= args.warmup:
            times.append(dt)

    # Accuracy. The reference is a full-precision run of the SAME contraction
    # with the SAME seed, written to node-local NVMe; comparing against it
    # isolates arithmetic error from every other difference.
    rel_err = None
    if args.ref_in:
        ref = np.load(args.ref_in, mmap_mode="r")
        # Accumulate over the leading axis: `out - ref` in one go would
        # allocate another full-size array (36.5 GiB at N=800).
        num2 = den2 = 0.0
        for i in range(result.shape[0]):
            r = np.asarray(ref[i])
            d = (result[i] - r).ravel()
            num2 += float(d @ d)
            rr = r.ravel()
            den2 += float(rr @ rr)
        rel_err = (num2 ** 0.5) / (den2 ** 0.5) if den2 else float("nan")
        if not den2:
            print("WARNING: reference has zero norm -- it was probably "
                  "written from the unused `out` array", flush=True)
        print(f"rel_err vs {args.ref_in}: {rel_err:.3e}", flush=True)
        del ref
    if args.ref_out:
        t = time.perf_counter()
        np.save(args.ref_out, result)
        print(f"wrote reference {args.ref_out} in {time.perf_counter()-t:.1f}s",
              flush=True)

    # Peak RSS bounds what their 480 GB GH200 could not fit. They report N=1100
    # (248 GB of operands) but mark N=1200 (310 GB) n.c., so PyBEST's true peak
    # is 1.55-1.94x the operand sum -- batching temporaries, not just operands.
    # Measuring it turns that bracket into a number.
    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20

    res = {"nbasis": args.nbasis, "nocc": args.nocc,
           "nvirt": args.nbasis - args.nocc,
           "nchol": args.nvec_factor * args.nbasis,
           "nvec_factor": args.nvec_factor,
           "backend": gpu_backend_select, "c_split": c_split, "x_split": x_split,
           "parts": args.parts, "subs": subs, "times_sec": times,
           "median_sec": statistics.median(times),
           "mean_sec": statistics.fmean(times),
           "total_gib": total, "host": platform.node(),
           "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "peak_rss_gib": peak_gib, "peak_over_operands": peak_gib / total,
           "cpus_visible": os.cpu_count(),
           "random": args.random, "seed": args.seed, "rel_err": rel_err,
           # The cuBLAS emulation knobs in force, recorded so a JSON result is
           # self-describing rather than depending on the job script.
           "cublas_env": {k: v for k, v in os.environ.items()
                          if k.startswith("CUBLAS_")}}
    print(f"\nmedian {res['median_sec']:.2f} s   mean {res['mean_sec']:.2f} s")
    print(f"peak_rss {peak_gib:.1f} GiB   = {peak_gib/total:.2f}x operand sum "
          f"({total:.1f} GiB)")
    path = os.environ.get(
        "BENCH_OUT",
        f"table4_N{args.nbasis}_o{args.nocc}_v{args.nvec_factor}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
