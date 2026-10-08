#!/bin/bash
# PyTorch's pinning fix at the sizes where PyTorch actually fails.
#
# Settled by now: the PyTorch transfer path is the defect worth fixing. CuPy
# already stages through a pinned pool, so a bounded pinned buffer does nothing
# for it (job 161877, ladder N=800: 258.12 -> 258.19 s, +1.0 GiB). PyTorch never
# pins, so every device-to-host copy is synchronous, and the cost grows with N:
#   N=800   275.83 -> 228.45 s  (-17.2%)      job 161368
#   N=1100 1473.39 -> 1030.72 s (-30.0%)      job 161368
# Beyond that PyTorch falls BELOW H100 (N=1200 0.96x, N=1300 0.92x), which is
# the result this job is meant to overturn.
#
# Two patches, both free:
#   pinned      as_numpy/get_numpy via one reused pinned staging buffer
#   cachedvram  freeze memory_usage(), worth -12.5% for PyTorch at N=1200
# so `both` is the candidate optimum. Cell 1 screens all four at N=800, where a
# cell is ~9 min, to confirm the two gains compose. The large N then run `both`
# only -- a paired baseline at N=1300 would cost another 97 min of GPU time, and
# the existing references are 1948.04 s (N=1200) and 2909.81 s (N=1300), with
# cachedvram-only already measured at 1704.41 s for N=1200.
#
# Everything that is compared within a size sits in THIS job: identical ladder
# cells have drifted 8.8% between jobs (258.12 vs 237.20 s at N=800 with
# identical peak_rss), so cross-job deltas under ~10% mean nothing.
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

cell () {   # cell <N> <strategy>
  local N=$1 STRAT=$2
  local TAG="ptpin_ladder_N${N}_${STRAT}" RAW
  RAW=$L/raw_ptpin_ladder_N${N}_${STRAT}.log
  echo "=============== $TAG ==============="
  nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used \
             --format=csv,noheader -l 30 > "$L/gpu_$TAG.csv" 2>/dev/null &
  local SMI=$!
  BENCH_OUT=$R/$TAG.json \
  $A exec --nv -B /data1 \
     --env PYBEST_PYTORCH_AVAIL=1,PYBEST_C_SPLITTING=1,BENCH_OUT=$R/$TAG.json \
     "$SIF" python $P/pin_experiment.py \
        --driver $P/repro_table4.py --strategy "$STRAT" \
        -- --nbasis "$N" --nocc 100 --nvec-factor 5 --reps 1 --warmup 1 \
        > "$RAW" 2>&1
  local rc=$?
  kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
  grep -aE "^(# |backend=|allocated|  (warmup|timed)|median|peak_rss)" "$RAW"
  echo "rc=$rc"
  [ $rc -ne 0 ] && tail -15 "$RAW"
  awk -F, '{gsub(/ |%/,"",$2); gsub(/ |%/,"",$3);
            if ($2+0>g) g=$2+0; if ($3+0>m) m=$3+0}
           END{print "gpu_max_util="g"%  mem_ctrl_max_util="m"%"}' "$L/gpu_$TAG.csv"
  echo
}

# 1. strategy screen, cheap size, fully paired
for STRAT in baseline pinned cachedvram both; do cell 800 "$STRAT"; done

# 2. the sizes where PyTorch loses to H100
cell 1200 pinned
cell 1200 both
cell 1300 both

echo "ALL DONE $(date)"
