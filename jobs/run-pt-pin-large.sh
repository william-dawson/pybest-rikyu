#!/bin/bash
# PyTorch pinning at N=1200 and N=1300: the two ladder sizes where PyTorch falls
# BELOW H100 in the published data (ours 1948.04 s vs their H100 C-split 1864.7;
# 2909.81 vs 2679.1), which is the result this is meant to overturn.
#
# Pinned only, no in-job baseline. The baselines exist from job 157159 under the
# same script and settings -- 1948.04 s and 2909.81 s -- and pairing them here
# would cost another 2.7 h of GPU time. Cross-job drift on identical ladder cells
# has reached 8.8%, so that is the uncertainty on these two deltas; the effect
# being chased is 20-30%, and it is anchored by two properly paired cells at
# smaller N: 286.52 -> 233.06 s at N=800 (-18.6%, job 161883) and
# 1473.39 -> 1030.72 s at N=1100 (-30.0%, job 161368).
#
# This replaces the `both` cells of job 161883, which tested pinning combined
# with a frozen free-VRAM reading. That combination is withdrawn: freezing does
# not remove overhead (memory_usage is one cudaMemGetInfo called five times), it
# changes the batch plan, and sizing batches against the largest reading a run
# ever sees is not a change to ship on a memory-constrained code.
#
# N=1300 peaks at 589.9 GiB host RSS unpinned, so --gpus=2 (800 GB) is required.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
R=$P/results; mkdir -p "$R"
L=$P/logs;   mkdir -p "$L"

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
free -g | head -2
echo

for N in 1200 1300; do
  TAG="ptpin_ladder_N${N}_pinned"; RAW=$L/raw_$TAG.log
  echo "=============== $TAG ==============="
  nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used \
             --format=csv,noheader -l 30 > "$L/gpu_$TAG.csv" 2>/dev/null &
  SMI=$!
  BENCH_OUT=$R/$TAG.json \
  $A exec --nv -B /data1 \
     --env PYBEST_PYTORCH_AVAIL=1,PYBEST_C_SPLITTING=1,BENCH_OUT=$R/$TAG.json \
     "$SIF" python $P/pin_experiment.py \
        --driver $P/repro_table4.py --strategy pinned \
        -- --nbasis "$N" --nocc 100 --nvec-factor 5 --reps 1 --warmup 1 \
        > "$RAW" 2>&1
  rc=$?
  kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
  grep -aE "^(# |backend=|allocated|  (warmup|timed)|median|peak_rss)" "$RAW"
  echo "rc=$rc"
  [ $rc -ne 0 ] && tail -15 "$RAW"
  awk -F, '{gsub(/ |%/,"",$2); gsub(/ |%/,"",$3);
            if ($2+0>g) g=$2+0; if ($3+0>m) m=$3+0}
           END{print "gpu_max_util="g"%  mem_ctrl_max_util="m"%"}' "$L/gpu_$TAG.csv"
  echo
done

echo "ALL DONE $(date)"
