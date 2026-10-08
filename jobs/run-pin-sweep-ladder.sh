#!/bin/bash
# Does the pinning fix carry over to the synthetic ladder -- the contraction
# the paper's Table 4 measures, and where PyTorch regressed BELOW H100?
#
# Measured at 240 AO molecular (job 161361), controlled in-job comparison:
#   PyTorch  CC 97.62 -> 61.29 s (-37%),  GPU: Generic 60.0 -> 24.7 s (-59%)
# The CuPy comparison there was confounded (its baseline took the dump_cache
# path), so this job provides clean baselines for both backends.
#
# Both patches reach this code path: c_splitting calls clean_memory at 37 sites
# and returns results through move_tensor_to_cpu -> _get_ops()["get_numpy"].
#
# Clean references, nocc=100, nvec=5N, C-split, --gpus=2:
#   N=800   CuPy  235.18 s   PyTorch  267.78 s   (GH200 306.6 / 355.9)
#   N=1100  CuPy 1046.47 s   PyTorch 1369.49 s   (GH200 1282.7 / 1401.7)
# N=1100 is the interesting one: PyTorch's advantage collapsed to 1.02x there,
# and at N=1200-1300 it fell below H100 outright.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
R=$P/results; mkdir -p "$R"
L=$P/logs;   mkdir -p "$L"

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
nvidia-smi --query-gpu=index,name --format=csv,noheader
free -g | sed -n 2p
echo

for N in 800 1100; do
  for BACKEND in cupy pytorch; do
    if [ "$BACKEND" = cupy ]; then ENVV=PYBEST_CUPY_AVAIL=1
    else ENVV=PYBEST_PYTORCH_AVAIL=1; fi
    for STRAT in baseline pinned; do
      TAG="pinladder_N${N}_${BACKEND}_${STRAT}"
      RAW=$L/raw_$TAG.log
      echo "=============== $TAG ==============="

      nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used \
                 --format=csv,noheader -l 30 > "$L/gpu_$TAG.csv" 2>/dev/null &
      SMI=$!

      BENCH_OUT=$R/$TAG.json \
      $A exec --nv -B /data1 \
         --env ${ENVV},PYBEST_C_SPLITTING=1,BENCH_OUT=$R/$TAG.json \
         "$SIF" python $P/pin_experiment.py \
            --driver $P/repro_table4.py --strategy $STRAT \
            -- --nbasis "$N" --nocc 100 --nvec-factor 5 --reps 1 --warmup 1 \
            --log-batching > "$RAW" 2>&1
      rc=$?

      kill $SMI 2>/dev/null; wait $SMI 2>/dev/null

      grep -aE "^(# |backend=|allocated|#batch|  (warmup|timed)|median|peak_rss|wrote|REFUSING|WARNING)" "$RAW"
      echo "rc=$rc"
      [ $rc -ne 0 ] && tail -20 "$RAW"
      awk -F, '{gsub(/ |%/,"",$2); gsub(/ |%/,"",$3);
                if ($2+0>g) g=$2+0; if ($3+0>m) m=$3+0}
               END{print "gpu_max_util="g"%  mem_ctrl_max_util="m"%"}' "$L/gpu_$TAG.csv"
      echo
    done
  done
done
echo "ALL DONE $(date)"
