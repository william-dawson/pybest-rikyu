#!/usr/bin/env python3
"""Measure cuBLAS floating-point emulation on GB200, before wiring it into PyBEST.

cuBLAS (CUDA 13.0 Update 2+) emulates GEMM on tensor cores in two distinct ways.
They are NOT the same mechanism and are easy to conflate:

  FP64 emulation   fixed-point, Ozaki-I / Ozaki-II. Splits each FP64 into a
                   shared power-of-2 exponent plus an M-bit integer mantissa.
                   CC 8.x/9.0/10.x/11.0/12.x. THIS is what a DGEMM uses.
  FP32 emulation   BF16x9. Represents each FP32 as three BF16 values,
                   a = a0 + 2^-8 a1 + 2^-16 a2, hence nine BF16 products.
                   CC 10.0/10.3 only. Applies to SGEMM, not DGEMM.

Everything is reachable through environment variables, so nothing in PyBEST has
to change -- CuPy and PyTorch call cublasDgemm and cuBLAS decides underneath:

  CUBLAS_EMULATE_DOUBLE_PRECISION            1/0   fixed-point FP64
  CUBLAS_EMULATE_SINGLE_PRECISION            1/0   BF16x9 FP32
  CUBLAS_EMULATION_STRATEGY                  performant | eager
  CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT  N   also selects FIXED mode

Mantissa bits are the accuracy dial. The Ozaki scheme needs >=53 bits to
guarantee native-FP64 accuracy; below that you are trading digits for speed.
NVIDIA report on Quantum Espresso: ADP 1.5x, 55 bits faster still, 39 bits
~3x with 12 significant digits retained. On GB200 they quote up to 2.3x for
DGEMM under ADP. Those are the numbers this probe should be checked against.

Two cautions this probe is built around:

* cuBLAS reads these at handle creation, so the env must be set before the
  library initialises. Every mode therefore runs in a FRESH SUBPROCESS.
* CuPy and PyTorch ship their own cuBLAS. They may be different versions, and
  only one of them may support emulation. Both are measured separately.

Accuracy uses random normal operands against a CPU float64 reference. Constant
fills (as repro_table4.py uses, correctly, for timing) have no cancellation and
would make every emulated mode look exact.

  apptainer exec --nv -B /data1 pybest-rikyu-v3.sif python precision_probe.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

N_ACC = 2048       # accuracy: CPU float64 reference stays sub-second
N_TIME = 8192      # timing: large enough to be tensor-core bound
REPS = 3

# (label, env). The env is applied before the child imports anything.
MODES = [
    ("fp64 native",              {"CUBLAS_EMULATE_DOUBLE_PRECISION": "0"}),
    ("fp64 emulated (ADP)",      {"CUBLAS_EMULATE_DOUBLE_PRECISION": "1"}),
    ("fp64 emulated, performant", {"CUBLAS_EMULATE_DOUBLE_PRECISION": "1",
                                   "CUBLAS_EMULATION_STRATEGY": "performant"}),
    ("fp64 emulated, 55 bits",   {"CUBLAS_EMULATE_DOUBLE_PRECISION": "1",
                                  "CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT": "55"}),
    ("fp64 emulated, 47 bits",   {"CUBLAS_EMULATE_DOUBLE_PRECISION": "1",
                                  "CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT": "47"}),
    ("fp64 emulated, 39 bits",   {"CUBLAS_EMULATE_DOUBLE_PRECISION": "1",
                                  "CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT": "39"}),
    ("fp64 emulated, 31 bits",   {"CUBLAS_EMULATE_DOUBLE_PRECISION": "1",
                                  "CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT": "31"}),
    ("fp32 native",              {"CUBLAS_EMULATE_SINGLE_PRECISION": "0"}),
    ("fp32 emulated (BF16x9)",   {"CUBLAS_EMULATE_SINGLE_PRECISION": "1"}),
]


def child(backend: str) -> int:
    """One measurement in this process, with the env we were launched with."""
    import numpy as np

    single = os.environ.get("PROBE_SINGLE") == "1"
    rng = np.random.default_rng(0)                 # identical data every mode
    a64 = rng.standard_normal((N_ACC, N_ACC))
    b64 = rng.standard_normal((N_ACC, N_ACC))
    ref = a64 @ b64                                # CPU float64: common yardstick

    if backend == "cupy":
        import cupy as xp
        to_dev = lambda x, d: xp.asarray(x, dtype=d)   # noqa: E731
        sync = xp.cuda.Stream.null.synchronize
        f32, f64 = xp.float32, xp.float64
        to_host = xp.asnumpy
        ver = xp.cuda.runtime.runtimeGetVersion()
        rand = lambda n, d: xp.asarray(rng.standard_normal((n, n)), dtype=d)  # noqa: E731
        free = lambda: xp.get_default_memory_pool().free_all_blocks()  # noqa: E731
    else:
        import torch
        to_dev = lambda x, d: torch.as_tensor(x, dtype=d).cuda()  # noqa: E731
        sync = torch.cuda.synchronize
        f32, f64 = torch.float32, torch.float64
        to_host = lambda t: t.cpu().numpy()  # noqa: E731
        ver = torch.version.cuda
        rand = lambda n, d: torch.as_tensor(  # noqa: E731
            rng.standard_normal((n, n)), dtype=d).cuda()
        free = torch.cuda.empty_cache

    dt = f32 if single else f64
    a, b = to_dev(a64, dt), to_dev(b64, dt)
    c = a @ b
    sync()
    got = to_host(c).astype(np.float64)
    rel_err = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    del a, b, c
    free()

    at, bt = rand(N_TIME, dt), rand(N_TIME, dt)
    for _ in range(2):
        at @ bt
    sync()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        at @ bt
        sync()
        ts.append(time.perf_counter() - t0)
    sec = min(ts)

    print(json.dumps({"sec": sec, "tflops": 2.0 * N_TIME ** 3 / sec / 1e12,
                      "rel_err": rel_err, "cuda": str(ver)}))
    return 0


def run(backend: str, env_extra: dict, single: bool) -> dict:
    env = dict(os.environ)
    env.update(env_extra)
    env["PROBE_SINGLE"] = "1" if single else "0"
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", backend],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return {"error": r.stderr.strip().splitlines()[-1][:120] if r.stderr else "failed"}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": f"unparseable: {r.stdout.strip()[-120:]}"}


def main() -> int:
    if "--child" in sys.argv:
        return child(sys.argv[sys.argv.index("--child") + 1])

    for backend in ("cupy", "pytorch"):
        print(f"\n{'='*72}\n{backend}\n{'='*72}")
        base = None
        print(f"{'mode':<30}{'TFLOP/s':>10}{'sec':>8}{'rel_err':>12}{'vs native':>11}")
        for label, env in MODES:
            single = "SINGLE" in "".join(env)
            r = run(backend, env, single)
            if "error" in r:
                print(f"{label:<30}{'':>10}{'':>8}{'':>12}  ERROR {r['error'][:40]}")
                continue
            if label == "fp64 native":
                base = r
            speed = f"{r['tflops']/base['tflops']:.2f}x" if base else "--"
            print(f"{label:<30}{r['tflops']:>10.2f}{r['sec']:>8.3f}"
                  f"{r['rel_err']:>12.2e}{speed:>11}")
    print("\nrel_err is against a CPU float64 reference, so the native FP64 row"
          "\nis the noise floor, not zero. Emulated rows should be read against it."
          "\nA mode whose time AND error both match native did nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
