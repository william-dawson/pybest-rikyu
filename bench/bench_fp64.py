#!/usr/bin/env python3
"""FP64 ceiling microbenchmark: one B200 vs Grace cores, plus H<->D bandwidth.

Run BEFORE any end-to-end PyBEST timing. Its job is to establish the hardware
ceiling so that a shortfall later is attributable to PyBEST's offload efficiency
rather than to the GPU's FP64 characteristics.

Facility figures imply ~40 TFLOP/s FP64 per B200 (64.160 PFLOPS / 1600 GPUs);
32 Neoverse-V2 cores are order 1-2 TFLOP/s. We measure rather than assume.

  A=/shared/software/apptainer/bin/apptainer
  $A exec --nv -B /data1 pybest-rikyu-v3.sif python bench_fp64.py
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import time

import numpy as np

REPS, WARMUP = 5, 2
GEMM_N = [2048, 4096, 8192, 16384]
# Cholesky-shaped tensordot mirroring PyBEST's "xac,xbd" operand layout.
TD_SHAPES = [(256, 64, 256), (512, 96, 384), (1024, 128, 512)]


def _time(fn, sync=None) -> float:
    """Median wall time over REPS after WARMUP, with device sync inside timing."""
    for _ in range(WARMUP):
        fn()
        if sync:
            sync()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def bench_cpu():
    out = []
    for n in GEMM_N:
        a, b = np.random.rand(n, n), np.random.rand(n, n)
        t = _time(lambda: a @ b)
        f = 2.0 * n**3
        out.append({"kind": "gemm", "n": n, "sec": t, "tflops": f / t / 1e12})
        print(f"  cpu  gemm n={n:<6} {t*1e3:9.2f} ms  {f/t/1e12:7.3f} TFLOP/s")
    for nx, no, nv in TD_SHAPES:
        x, y = np.random.rand(nx, no, nv), np.random.rand(nx, no, nv)
        t = _time(lambda: np.tensordot(x, y, axes=([0], [0])))
        f = 2.0 * nx * (no * nv) ** 2
        out.append({"kind": "tensordot", "shape": [nx, no, nv], "sec": t,
                    "tflops": f / t / 1e12})
        print(f"  cpu  td   {nx}x{no}x{nv:<5} {t*1e3:9.2f} ms  {f/t/1e12:7.3f} TFLOP/s")
    return out


def bench_gpu():
    try:
        import cupy as cp
        cp.zeros(1)
    except Exception as exc:  # noqa: BLE001
        print(f"  cupy unavailable / no device: {exc}")
        return None

    sync = cp.cuda.Stream.null.synchronize
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"  device: {total/2**30:.1f} GiB total, {free/2**30:.1f} GiB free")
    out = []
    for n in GEMM_N:
        a = cp.random.rand(n, n, dtype=cp.float64)
        b = cp.random.rand(n, n, dtype=cp.float64)
        t = _time(lambda: a @ b, sync)
        f = 2.0 * n**3
        out.append({"kind": "gemm", "n": n, "sec": t, "tflops": f / t / 1e12})
        print(f"  gpu  gemm n={n:<6} {t*1e3:9.2f} ms  {f/t/1e12:7.3f} TFLOP/s")
        del a, b
        cp.get_default_memory_pool().free_all_blocks()
    for nx, no, nv in TD_SHAPES:
        x = cp.random.rand(nx, no, nv, dtype=cp.float64)
        y = cp.random.rand(nx, no, nv, dtype=cp.float64)
        t = _time(lambda: cp.tensordot(x, y, axes=([0], [0])), sync)
        f = 2.0 * nx * (no * nv) ** 2
        out.append({"kind": "tensordot", "shape": [nx, no, nv], "sec": t,
                    "tflops": f / t / 1e12})
        print(f"  gpu  td   {nx}x{no}x{nv:<5} {t*1e3:9.2f} ms  {f/t/1e12:7.3f} TFLOP/s")
        del x, y
        cp.get_default_memory_pool().free_all_blocks()

    # PyBEST offloads by copying operands per call. On PCIe boxes that transfer
    # cost is what defeats naive offload; NVLink-C2C is cache-coherent at a
    # claimed 450 GB/s, so check what we actually observe.
    for mb in (64, 512, 2048):
        host = np.random.rand(mb * 2**20 // 8)
        nb = host.nbytes
        t_h2d = _time(lambda: cp.asarray(host), sync)
        dev = cp.asarray(host)
        t_d2h = _time(lambda: cp.asnumpy(dev), sync)
        out.append({"kind": "transfer", "mb": mb, "h2d_gbs": nb / t_h2d / 1e9,
                    "d2h_gbs": nb / t_d2h / 1e9})
        print(f"  xfer {mb:>5} MiB  H2D {nb/t_h2d/1e9:7.1f} GB/s   "
              f"D2H {nb/t_d2h/1e9:7.1f} GB/s")
        del dev
        cp.get_default_memory_pool().free_all_blocks()
    return out


def main() -> int:
    env = {k: os.environ.get(k) for k in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "SLURM_JOB_ID",
        "SLURM_CPUS_ON_NODE", "PYBEST_CUPY_AVAIL")}
    print(f"host={platform.node()} python={platform.python_version()} "
          f"numpy={np.__version__}")
    print(f"cpus_visible={os.cpu_count()} env={env}")
    print("\n-- CPU (numpy / bundled OpenBLAS) --")
    cpu = bench_cpu()
    print("\n-- GPU (CuPy, one B200) --")
    gpu = bench_gpu()

    if gpu:
        cg = max(r["tflops"] for r in cpu if r["kind"] == "gemm")
        gg = max(r["tflops"] for r in gpu if r["kind"] == "gemm")
        print(f"\npeak FP64 GEMM: cpu {cg:.2f} / gpu {gg:.2f} TFLOP/s "
              f"-> speedup {gg/cg:.1f}x")
        print("This ratio is the CEILING. Any smaller end-to-end PyBEST speedup "
              "is offload overhead, not hardware.")

    path = os.environ.get("BENCH_OUT", "bench_fp64.json")
    with open(path, "w") as fh:
        json.dump({"cpu": cpu, "gpu": gpu, "env": env, "host": platform.node(),
                   "cpus_visible": os.cpu_count()}, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
