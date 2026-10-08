import pybest, numpy, scipy, opt_einsum
print("PYBEST", pybest.__version__, "| numpy", numpy.__version__,
      "| scipy", scipy.__version__, "| opt_einsum", opt_einsum.__version__)
from pybest.gbasis.cholesky_eri import PYBEST_CHOLESKY_ENABLED
print("CHOLESKY_ENABLED", PYBEST_CHOLESKY_ENABLED)
from pybest.linalg._gpu_support import gpu_contraction_optimized as g
print("GPU_PATTERNS", len(g))

import cupy as cp
x = cp.random.rand(2048, 2048, dtype=cp.float64); y = x @ x
cp.cuda.Stream.null.synchronize()
f, t = cp.cuda.runtime.memGetInfo()
print(f"CUPY {cp.__version__} dgemm_ok VRAM {t/2**30:.1f} GiB total {f/2**30:.1f} free")

import torch
print("TORCH", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "devices", torch.cuda.device_count())
if torch.cuda.is_available():
    print("TORCH_DEV", torch.cuda.get_device_name(0))
    a = torch.rand(2048, 2048, dtype=torch.float64, device="cuda")
    b = a @ a; torch.cuda.synchronize()
    print("TORCH_DGEMM_OK trace=", float(torch.trace(b)))

# the two backends PyBEST actually dispatches on, with env set
import os
print("ENV_CUPY", os.environ.get("PYBEST_CUPY_AVAIL"),
      "ENV_TORCH", os.environ.get("PYBEST_PYTORCH_AVAIL"),
      "C_SPLIT", os.environ.get("PYBEST_C_SPLITTING"))
