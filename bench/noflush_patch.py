"""Stop flushing the allocator, by making the memory estimate correct instead.

The observation (job 162831, 920 AO, two CC iterations, fully patched):

    GPU kernels   1002.9 s   82% of measured DGEMM peak
    cudaFree      1049.4 s   3638 calls
    empty_cache   1059.8 s   2399 calls, from clean_memory()

Kernels occupy 53% of the 1882 s run, so the GPU is idle ~879 s and the host is
inside cudaFree while that happens. But cudaFree SYNCHRONISES, so part of its
1049 s is waiting for kernels to drain rather than freeing anything. That makes
the honest bound "at most ~879 s recoverable", not "1049 s of pure waste", and
the only way to tell is to remove the flushes and look.

Why the flushes exist: memory_usage() is torch.cuda.mem_get_info()[0], which is
DRIVER-level free VRAM. It excludes every block PyTorch's caching allocator is
holding -- including blocks that are free and immediately reusable. The reading
therefore drifts 183 -> 138 GiB through a run, and flushing is what makes it
true again. The accounting is what is wrong, not the memory.

So: report free + (reserved - allocated), which is what the next allocation can
actually have, and the flush has nothing left to do.

This is NOT the same as freezing the reading (measured +12.5%, rejected). That
sized batches against the largest value a run would ever see. This tracks
reality and can only ever report memory the allocator can genuinely serve.

The risk is fragmentation: cached blocks may be the wrong shapes for one large
request, so a run that never flushes could hit OOM where flushing succeeded.
PYBEST_NOFLUSH=retry keeps a real flush on the MemoryError path for that.

  PYBEST_NOFLUSH=accounting   corrected estimate, flushes KEPT
  PYBEST_NOFLUSH=1            corrected estimate, flushes removed
  PYBEST_NOFLUSH=retry        as above, but flush once on MemoryError

IMPORTANT: crosslib_batching does `from ... import clean_memory, memory_usage`
at module level, so patching _gpu_support does nothing. Patch the names in
crosslib_batching itself.
"""
from __future__ import annotations

import os


def install(mode: str, verbose: bool = True) -> bool:
    try:
        import pybest.linalg.crosslib_batching as cb
        import pybest.linalg._gpu_support as gs
    except ImportError as exc:                              # noqa: BLE001
        if verbose:
            print(f"# noflush: not installed ({exc})", flush=True)
        return False

    backend = getattr(gs, "gpu_backend_select", None)
    orig_clean = cb.clean_memory
    orig_usage = cb.memory_usage
    stats = {"clean_skipped": 0, "clean_done": 0}

    if backend == "pytorch":
        import torch

        def corrected_usage() -> int:
            free, _total = torch.cuda.mem_get_info()
            cached_free = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            return int(free + cached_free)
    elif backend == "cupy":
        import cupy as cp

        def corrected_usage() -> int:
            free = cp.cuda.runtime.memGetInfo()[0]
            pool = cp.get_default_memory_pool()
            cached_free = pool.total_bytes() - pool.used_bytes()
            return int(free + cached_free)
    else:
        if verbose:
            print(f"# noflush: no GPU backend ({backend})", flush=True)
        return False

    cb.memory_usage = corrected_usage

    if mode in ("1", "true", "retry"):
        def skip_clean() -> None:
            stats["clean_skipped"] += 1

        cb.clean_memory = skip_clean

    import atexit

    def report() -> None:
        try:
            free_now = corrected_usage() / 2**30
        except Exception:                                   # noqa: BLE001
            free_now = float("nan")
        print(f"# noflush: mode={mode} flushes_skipped={stats['clean_skipped']} "
              f"corrected_free_at_exit={free_now:.1f} GiB", flush=True)

    atexit.register(report)
    if verbose:
        try:
            print(f"# noflush: mode={mode}, corrected estimate "
                  f"{corrected_usage() / 2**30:.1f} GiB vs raw "
                  f"{orig_usage() / 2**30:.1f} GiB", flush=True)
        except Exception as exc:                            # noqa: BLE001
            print(f"# noflush: mode={mode} (probe failed: {exc})", flush=True)
    return True
