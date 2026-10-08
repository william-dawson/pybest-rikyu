#!/bin/bash
# Price RCCSD: unravel and its two avoidable costs. CPU only -- no GPU is used,
# the allocation just buys cores and host memory.
#
# Why this is the next target: after the pinning fix, the Nsight trace of a
# pinned 240 AO run puts GEMM kernels at 3.30 s, all transfers at 1.77 s,
# cudaMalloc+cudaFree at 8.64 s, and ~52 s of 66.6 s in host-side Python/numpy.
# unravel is 462 s of the 920 AO run, reproduces to within 1% between backends,
# and touches no GPU -- and unlike the allocator churn, changing it perturbs
# nothing: no batching decision, no memory estimate, no results.
#
# Two claims under test, both read off dense_four_index.py:
#   assign_triu (line 490) builds np.triu_indices(nacto*nactv) per call. At
#     920 AO that is 9.69 GiB of int64 index arrays plus a 605 M-element
#     scatter, where contiguous row slices would do.
#   iadd_transpose (line 828) is `a[:] = a + a.transpose(2,3,0,1)*f`, two full
#     temporaries; and on an (o,v,o,v) array that permutation is exactly matrix
#     transpose on the (ov,ov) view, so it is M += M.T, blockable in place.
# The script asserts the fast forms are BITWISE equal, not approximate.
#
# Memory: 920 AO needs t_2 at 9.69 GiB plus ~29 GiB of temporaries for the slow
# path, so --gpus=1 (400 GB) is ample.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
R=$P/results/unravel; mkdir -p "$R"
L=$P/logs; mkdir -p "$L"

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
free -g | head -2
echo

# Block size matters for the in-place symmetrisation: too small and Python loop
# overhead dominates, too large and the block temporary leaves cache.
for BLK in 1024 4096; do
  TAG="unravel_blk${BLK}"
  echo "=============== $TAG ==============="
  BENCH_OUT=$R/$TAG.json UC_BLOCK=$BLK \
  $A exec -B /data1 \
     --env UC_BLOCK=$BLK,BENCH_OUT=$R/$TAG.json \
     "$SIF" python $P/unravel_cost.py 2>&1 | tee "$L/raw_$TAG.log"
  echo "rc=${PIPESTATUS[0]}"
  echo
done

echo "ALL DONE $(date)"
