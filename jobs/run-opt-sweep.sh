#!/bin/bash
# Two more free optimisations on the ladder, both monkey-patched so the
# read-only image is untouched.
#
# A) cachedvram -- query free VRAM once instead of per call.
#    At N=1200 the reading falls 183.3 -> 138.1 GiB between passes, and the
#    second pass runs 5.7% slower (CuPy) / 18.0% slower (PyTorch) on identical
#    input. Clean N=1200 references from job 157159:
#        CuPy    warmup 1534.52  timed 1622.32   (+5.7%)
#        PyTorch warmup 1650.51  timed 1948.04   (+18.0%)
#    If caching flattens warmup-to-timed, the drift mechanism is confirmed and
#    a one-line change in their code recovers it.
#
# B) fixed parts -- c_splitting lines 904-906 take parts_a/parts_c/parts_b from
#    a `parts` kwarg, overriding the heuristic result entirely, which also
#    bypasses the `ecfd < 0.4 * memhave` cliff. Swept at N=800, where a pass is
#    cheap, against the clean references CuPy 235.18 s / PyTorch 267.78 s.
#
# Nothing here changes peak memory except through the batch granularity itself,
# which is the point of sweeping it.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
R=$P/results; mkdir -p "$R"
L=$P/logs;   mkdir -p "$L"

run_cell () {   # tag, backend, strategy, extra-driver-args
  local TAG=$1 BACKEND=$2 STRAT=$3; shift 3
  local ENVV RAW; RAW=$L/raw_$TAG.log
  if [ "$BACKEND" = cupy ]; then ENVV=PYBEST_CUPY_AVAIL=1
  else ENVV=PYBEST_PYTORCH_AVAIL=1; fi
  echo "=============== $TAG ==============="
  nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used \
             --format=csv,noheader -l 30 > "$L/gpu_$TAG.csv" 2>/dev/null &
  local SMI=$!
  BENCH_OUT=$R/$TAG.json \
  $A exec --nv -B /data1 \
     --env ${ENVV},PYBEST_C_SPLITTING=1,BENCH_OUT=$R/$TAG.json \
     "$SIF" python $P/pin_experiment.py \
        --driver $P/repro_table4.py --strategy "$STRAT" -- "$@" \
        > "$RAW" 2>&1
  local rc=$?
  kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
  grep -aE "^(# |backend=|#batch|  (warmup|timed)|median|peak_rss)" "$RAW"
  echo "rc=$rc"
  [ $rc -ne 0 ] && tail -15 "$RAW"
  awk -F, '{gsub(/ |%/,"",$2); gsub(/ |%/,"",$3);
            if ($2+0>g) g=$2+0; if ($3+0>m) m=$3+0}
           END{print "gpu_max_util="g"%  mem_ctrl_max_util="m"%"}' "$L/gpu_$TAG.csv"
  echo
}

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
echo

# A) the drift fix, where the drift is actually measurable
for BACKEND in cupy pytorch; do
  run_cell "opt_N1200_${BACKEND}_cachedvram" "$BACKEND" cachedvram \
     --nbasis 1200 --nocc 100 --nvec-factor 5 --reps 1 --warmup 1 --log-batching
done

# B) granularity, cheap size
for BACKEND in cupy pytorch; do
  for PARTS in 2 4 8; do
    run_cell "opt_N800_${BACKEND}_parts${PARTS}" "$BACKEND" baseline \
       --nbasis 800 --nocc 100 --nvec-factor 5 --reps 1 --warmup 1 \
       --log-batching --parts "$PARTS"
  done
done
echo "ALL DONE $(date)"
