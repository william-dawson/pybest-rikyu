#!/bin/bash
#SBATCH --job-name=t4-N800
#SBATCH --partition=gpu
#SBATCH --account=rkp00012
#SBATCH --gpus=2
#SBATCH --time=02:30:00
#SBATCH --chdir=/data1/rkp00012/rku00036/pybest
#SBATCH --output=/data1/rkp00012/rku00036/pybest/logs/t4-N800.out
#SBATCH --error=/data1/rkp00012/rku00036/pybest/logs/t4-N800.err
#
# Reproduce Table 4 (Dobrowolska et al., JCTC 2026, 22, 6533) at N=800,
# nocc=100, nvec=5N. GH200 C-split references: CuPy 306.6 s, PyTorch 355.9 s.
#
# --gpus=2 buys 64 cores + 800 GB: their GH200 runs used 72 CPU cores and the
# metric includes CPU-side batching prep, so core count is part of the
# comparison. Only 1 GPU is actually used (PyBEST is single-GPU).
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
R=$P/results; mkdir -p "$R"

echo "host=$(hostname) nproc=$(nproc) date=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
free -g | sed -n 2p

for BACKEND in cupy pytorch; do
  if [ "$BACKEND" = cupy ]; then ENVV="PYBEST_CUPY_AVAIL=1"; else ENVV="PYBEST_PYTORCH_AVAIL=1"; fi
  TAG="N800_o100_v5_${BACKEND}_Csplit"
  echo "=============== $TAG ==============="
  BENCH_OUT=$R/$TAG.json \
  $A exec --nv -B /data1 \
     --env ${ENVV},PYBEST_C_SPLITTING=1,BENCH_OUT=$R/$TAG.json \
     "$SIF" python $P/repro_table4.py --nbasis 800 --nocc 100 \
        --nvec-factor 5 --reps 4 --warmup 1 2>&1 \
    | grep -aE "^(N=|  (xac|xbd|ecfd|out|TOTAL)|backend=|WARNING|allocating|allocated|  (warmup|timed)|median|wrote|REFUSING)"
  echo "exit=$?"
done
echo "ALL DONE $(date)"
ls -la $R
