#!/usr/bin/env python3
"""What does a host<->device transfer actually cost on GB200, by mechanism?

Motivation, from the Nsight Systems trace of a 240 AO CCSD (job 161307):
H2D moved 260 GB at 128 GB/s out of ORDINARY PAGEABLE numpy memory, while D2H
moved 65.8 GB at 3.1 GB/s into ordinary pageable numpy memory. A 41x asymmetry
between directions, with the same kind of host buffer on both ends, is not
explained by the textbook "pageable copies are staged" story alone -- on a
Grace platform the GPU can also reach host memory coherently over NVLink-C2C,
and that path may be in play for one direction and not the other.

This settles what each mechanism is worth before any of them is wired into
PyBEST. No PyBEST, no CCSD: one buffer, five mechanisms, both directions.

  pageable    plain numpy. What PyBEST does today.
  staged      our current patch: a reused pinned buffer, then .copy() out of
              it, because the buffer is reused and the caller keeps the result.
  registered  cudaHostRegister on the numpy array itself, so the DMA engine
              writes the FINAL destination. No bounce, no second copy.
  managed     cudaMallocManaged. One pointer, migrated or coherently accessed
              by hardware; "transfer" becomes a touch.
  ats         plain malloc'd memory handed to the GPU directly, no registration
              at all. Only works where system-allocated memory is supported;
              expected to fail cleanly elsewhere, which is itself a result.

Also reports the ONE-OFF cost of cudaHostRegister per GiB, since a scheme that
registers PyBEST's long-lived arrays pays it once and a scheme that registers
per call pays it 1700 times -- which is exactly the mistake CuPy makes (1032
cudaHostAlloc calls, 12.74 s, in the same trace).

  apptainer exec --nv -B /data1 pybest-rikyu-v3.sif python transfer_modes.py
"""
from __future__ import annotations

import ctypes
import json
import os
import time

import numpy as np

GIB = 1 << 30
SIZE = int(os.environ.get("TM_GIB", "8")) * GIB   # bytes per buffer
REPS = int(os.environ.get("TM_REPS", "5"))
N = SIZE // 8                                     # float64 elements


def bw(nbytes: int, sec: float) -> float:
    return nbytes / sec / 1e9


def timeit(fn, reps: int = REPS) -> float:
    fn()                                  # warm up: first touch, JIT, pool fill
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)                        # best case: we want the ceiling


# --------------------------------------------------------------- cupy
def run_cupy() -> dict:
    import cupy as cp

    out: dict = {}
    sync = cp.cuda.Stream.null.synchronize
    dev = cp.empty(N, dtype=cp.float64)
    dev.fill(1.0)
    sync()

    # 1. pageable, both directions
    host = np.ones(N, dtype=np.float64)

    def d2h_pageable():
        cp.asnumpy(dev)
        sync()

    def h2d_pageable():
        cp.asarray(host)
        sync()

    out["d2h_pageable"] = bw(SIZE, timeit(d2h_pageable))
    out["h2d_pageable"] = bw(SIZE, timeit(h2d_pageable))

    # 2. staged through a reused pinned buffer, then copied out (our patch)
    mem = cp.cuda.alloc_pinned_memory(SIZE)
    staging = np.frombuffer(mem, dtype=np.float64, count=N)

    def d2h_staged():
        dev.get(out=staging)
        staging.copy()
        sync()

    out["d2h_staged"] = bw(SIZE, timeit(d2h_staged))

    # 2b. the same without the copy, to price the copy itself
    def d2h_pinned_nocopy():
        dev.get(out=staging)
        sync()

    out["d2h_pinned_nocopy"] = bw(SIZE, timeit(d2h_pinned_nocopy))
    del mem, staging

    # 3. cudaHostRegister on an ordinary numpy array: DMA to the real destination
    reg = np.empty(N, dtype=np.float64)
    ptr = reg.ctypes.data
    t0 = time.perf_counter()
    cp.cuda.runtime.hostRegister(ptr, SIZE, 0)
    out["register_sec_per_gib"] = (time.perf_counter() - t0) / (SIZE / GIB)
    try:
        def d2h_registered():
            dev.get(out=reg)
            sync()

        def h2d_registered():
            cp.asarray(reg)
            sync()

        out["d2h_registered"] = bw(SIZE, timeit(d2h_registered))
        out["h2d_registered"] = bw(SIZE, timeit(h2d_registered))
    finally:
        cp.cuda.runtime.hostUnregister(ptr)

    # 4. managed memory: no explicit transfer, the hardware decides
    try:
        mm = cp.cuda.malloc_managed(SIZE)
        man = cp.ndarray((N,), dtype=cp.float64, memptr=mm)
        man.fill(0.0)
        sync()

        def managed_gpu_write():
            man[:] = dev          # device kernel writes managed memory
            sync()

        def managed_cpu_read():
            # CPU-side sum forces the data host-side through coherence/migration
            float(np.frombuffer(                      # view, no copy
                (ctypes.c_double * 1024).from_address(int(man.data.ptr)),
                dtype=np.float64, count=1024).sum())

        out["managed_gpu_write"] = bw(SIZE, timeit(managed_gpu_write))
        managed_cpu_read()
        out["managed_cpu_touch_ok"] = True
        del man, mm
    except Exception as exc:                           # noqa: BLE001
        out["managed_error"] = f"{type(exc).__name__}: {exc}"[:160]

    # 5. system-allocated (ATS): hand the GPU a plain malloc'd pointer
    try:
        plain = np.ones(N, dtype=np.float64)
        unowned = cp.cuda.UnownedMemory(plain.ctypes.data, SIZE, plain)
        view = cp.ndarray((N,), dtype=cp.float64,
                          memptr=cp.cuda.MemoryPointer(unowned, 0))

        def ats_gpu_read():
            float(view.sum())      # kernel dereferences host memory directly
            sync()

        out["ats_gpu_read"] = bw(SIZE, timeit(ats_gpu_read, reps=2))
        out["ats_supported"] = True
    except Exception as exc:                           # noqa: BLE001
        out["ats_supported"] = False
        out["ats_error"] = f"{type(exc).__name__}: {exc}"[:160]

    return out


# ------------------------------------------------------------- pytorch
def run_torch() -> dict:
    import torch

    out: dict = {}
    sync = torch.cuda.synchronize
    dev = torch.ones(N, dtype=torch.float64, device="cuda")
    sync()

    host = np.ones(N, dtype=np.float64)

    def d2h_pageable():
        dev.cpu().numpy()          # exactly what PyBEST's as_numpy does
        sync()

    def h2d_pageable():
        torch.as_tensor(host).to("cuda")
        sync()

    out["d2h_pageable"] = bw(SIZE, timeit(d2h_pageable))
    out["h2d_pageable"] = bw(SIZE, timeit(h2d_pageable))

    pin = torch.empty(N, dtype=torch.float64, pin_memory=True)

    def d2h_staged():
        pin.copy_(dev, non_blocking=True)
        sync()
        pin.numpy().copy()

    def d2h_pinned_nocopy():
        pin.copy_(dev, non_blocking=True)
        sync()

    out["d2h_staged"] = bw(SIZE, timeit(d2h_staged))
    out["d2h_pinned_nocopy"] = bw(SIZE, timeit(d2h_pinned_nocopy))
    del pin

    # cudaHostRegister an ordinary numpy array, then wrap it with from_numpy:
    # the copy_ lands in PyBEST's own memory with no staging buffer.
    reg = np.empty(N, dtype=np.float64)
    ptr = reg.ctypes.data
    cudart = torch.cuda.cudart()
    t0 = time.perf_counter()
    rc = cudart.cudaHostRegister(ptr, SIZE, 0)
    out["register_sec_per_gib"] = (time.perf_counter() - t0) / (SIZE / GIB)
    out["register_rc"] = int(rc)
    try:
        tgt = torch.from_numpy(reg)

        def d2h_registered():
            tgt.copy_(dev, non_blocking=True)
            sync()

        def h2d_registered():
            torch.from_numpy(reg).to("cuda", non_blocking=True)
            sync()

        out["d2h_registered"] = bw(SIZE, timeit(d2h_registered))
        out["h2d_registered"] = bw(SIZE, timeit(h2d_registered))
    finally:
        cudart.cudaHostUnregister(ptr)

    return out


def main() -> int:
    res = {"buffer_gib": SIZE / GIB, "reps": REPS}
    for name, fn in (("cupy", run_cupy), ("pytorch", run_torch)):
        print(f"\n{'=' * 64}\n{name}\n{'=' * 64}", flush=True)
        try:
            r = fn()
        except Exception as exc:                       # noqa: BLE001
            r = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        res[name] = r
        for k, v in r.items():
            unit = "GB/s" if k.startswith(("d2h", "h2d", "managed_gpu", "ats_gpu")) else ""
            print(f"  {k:<24} {v if not isinstance(v, float) else round(v, 2)} {unit}",
                  flush=True)
    dest = os.environ.get("BENCH_OUT")
    if dest:
        with open(dest, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"\nwrote {dest}")
    print("\nThe numbers that matter: d2h_pageable is the defect, d2h_staged is our\n"
          "current patch, d2h_registered is the patch we did not write, and the gap\n"
          "between d2h_staged and d2h_pinned_nocopy is what the extra copy costs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
