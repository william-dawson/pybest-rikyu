#!/bin/bash
#SBATCH --job-name=t4-scale
#SBATCH --partition=gpu
#SBATCH --account=rkp00012
#SBATCH --gpus=2
#SBATCH --time=08:00:00
#SBATCH --chdir=/data1/rkp00012/rku00036/pybest
#SBATCH --output=/data1/rkp00012/rku00036/pybest/logs/t4-scale-%j.out
#SBATCH --error=/data1/rkp00012/rku00036/pybest/logs/t4-scale-%j.err
#
# Table 4 scaling series A/B/C: N=900,1000,1100 at nocc=100, nvec=5N, C-split,
# CuPy and PyTorch. Extends the N=800 point from job 147057 into a 4-point curve.
#
# Published GH200 C-split references (s):
#   N=900   CuPy 495.0   PyTorch 587.1
#   N=1000  CuPy 829.1   PyTorch 805.8
#   N=1100  CuPy 1282.7  PyTorch 1401.7
#
# --gpus=2 (64 cores) for EVERY N, matching the N=800 run. N=900 would fit in
# 1 GPU, but that yields only 32 cores and their metric includes CPU-side
# batching prep -- mixing core counts would confound N with core count.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer   # not on PATH under ssh 'cmd'
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
R=$P/results; mkdir -p "$R"
L=$P/logs;   mkdir -p "$L"

echo "host=$(hostname) nproc=$(nproc) job=$SLURM_JOB_ID date=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
free -g | sed -n 2p
echo

for N in 900 1000 1100; do
  for BACKEND in cupy pytorch; do
    if [ "$BACKEND" = cupy ]; then ENVV=PYBEST_CUPY_AVAIL=1
    else ENVV=PYBEST_PYTORCH_AVAIL=1; fi
    TAG="N${N}_o100_v5_${BACKEND}_Csplit"
    RAW=$L/raw_$TAG.log
    echo "=============== $TAG ==============="

    # Sample the GPU for the duration. AGENTS.md's central hazard is PyBEST
    # falling back to CPU silently, so every cell gets its own utilization
    # trace rather than relying on the env var being set.
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw \
               --format=csv,noheader -l 30 > "$L/gpu_$TAG.csv" 2>/dev/null &
    SMI=$!

    # Full output to RAW so a traceback is never lost; filter only for display.
    $A exec --nv -B /data1 \
       --env ${ENVV},PYBEST_C_SPLITTING=1,BENCH_OUT=$R/$TAG.json \
       "$SIF" python $P/repro_table4.py --nbasis "$N" --nocc 100 \
          --nvec-factor 5 --reps 4 --warmup 1 > "$RAW" 2>&1
    rc=$?                                   # python's status, not a pipeline's

    kill $SMI 2>/dev/null; wait $SMI 2>/dev/null

    grep -aE "^(N=|  (xac|xbd|ecfd|out|TOTAL)|backend=|WARNING|allocated|  (warmup|timed)|median|peak_rss|wrote|REFUSING)" "$RAW"
    echo "rc=$rc  raw=$RAW"
    if [ $rc -ne 0 ]; then
      echo "---- FAILED, last 30 lines of raw ----"
      tail -30 "$RAW"
    fi
    # GPU evidence: max utilization seen while this cell ran.
    awk -F, '{gsub(/ |%/,"",$2); if ($2+0>m) m=$2+0} END{print "gpu_max_util="m"%"}' \
        "$L/gpu_$TAG.csv"
    echo
  done
done
echo "ALL DONE $(date)"
ls -la $R
