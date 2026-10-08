#!/bin/bash
# Does removing the allocator flush actually recover the idle GPU time?
#
# Job 162831 at 920 AO, two CC iterations, fully patched:
#   GPU kernels  1002.9 s  (3.16e16 FLOP = 31.5 TFLOP/s = 82% of DGEMM peak)
#   cudaFree     1049.4 s  (3638 calls)
#   empty_cache  1059.8 s  (2399 calls, from clean_memory())
# Kernels occupy 53% of the 1882 s run, so the GPU is idle ~879 s with the host
# inside cudaFree. But cudaFree SYNCHRONISES, so an unknown part of its 1049 s
# is draining kernels rather than freeing. ~879 s is the ceiling on what can be
# recovered, not a prediction. This measures the actual figure.
#
# Three cells, paired in one job because 920 AO totals swing 15% between jobs:
#   flush       current behaviour, the control
#   accounting  corrected memory estimate, flushes KEPT -- isolates the effect
#               of the estimate alone on the batch plan
#   noflush     corrected estimate AND no flushes -- the proposal
#
# The corrected estimate is free + (reserved - allocated): memory the next
# allocation can genuinely have. That is NOT the rejected "freeze the reading"
# experiment, which sized batches against the largest value a run ever sees.
#
# An OOM in the noflush cell is a RESULT, not a failure -- fragmentation is the
# named risk -- so the script records it rather than treating it as an error.
#
# maxiter=2: we are measuring CCSD time, not reproducing the paper's metric.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
S=$P/systems
L=$P/logs; mkdir -p "$L"
W=/tmp/nofl-${SLURM_JOB_ID:-$$}
SYS=${SYSNAME:-h2o10_aug-cc-pvtz_ccsd}
BASE="PYBEST_PYTORCH_AVAIL=1,PYBEST_C_SPLITTING=1,PYBEST_D2H_VIEW=1,PYBEST_UNRAVEL_FIX=1,PYBEST_BENCH_DIR=$P,PYBEST_MAXITER=${MAXIT:-2}"

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "system=$SYS maxiter=${MAXIT:-2}"; echo

for MODE in flush accounting noflush; do
  case $MODE in
    flush)      EXTRA="" ;;
    accounting) EXTRA=",PYBEST_NOFLUSH=accounting" ;;
    noflush)    EXTRA=",PYBEST_NOFLUSH=1" ;;
  esac
  TAG="nofl_${SYS}_${MODE}"; RAW=$L/raw_$TAG.log
  TMPD=$W-$MODE; mkdir -p "$TMPD"
  echo "=============== $TAG ==============="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
             --format=csv,noheader -l 20 > "$L/gpu_$TAG.csv" 2>/dev/null &
  SMI=$!
  /usr/bin/time -v $A exec --nv -B /data1 -B /tmp \
     --env ${BASE},PYBEST_TEMP=$TMPD${EXTRA} \
     "$SIF" python $S/${SYS}.py > "$RAW" 2>&1
  rc=$?
  kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
  grep -aE "^(# |Total energy)" "$RAW"
  grep -aE "Maximum resident set size" "$RAW" || true
  sed -n '/Overview of CPU time usage/,/Page swaps/p' "$RAW" \
    | grep -aE "^(GPU: |Base: contract|RCCSD: unravel|Ints: CD-ERI|Total )" || true
  # OOM here is a result: fragmentation is the named risk of not flushing.
  if grep -aqE "OutOfMemoryError|CUDA out of memory|MemoryError" "$RAW"; then
    echo "RESULT: ran out of memory -- fragmentation risk realised"
    grep -aE "OutOfMemoryError|CUDA out of memory|MemoryError" "$RAW" | head -3
  fi
  # A patch that does not install is silent; say so loudly.
  for want in "d2h_view: view where" "unravel: assign_triu"; do
    grep -aqF "$want" "$RAW" || echo "PATCH DID NOT INSTALL: $want"
  done
  [ "$MODE" != flush ] && { grep -aqF "# noflush: mode=" "$RAW" \
    || echo "PATCH DID NOT INSTALL: noflush"; }
  echo "rc=$rc"
  [ $rc -ne 0 ] && tail -12 "$RAW"
  # Mean GPU utilisation: the point of the exercise is raising the duty cycle.
  awk -F, '{gsub(/ |%/,"",$2); s+=$2; n++} END{if(n) printf "gpu_mean_util=%.1f%% over %d samples\n", s/n, n}' \
      "$L/gpu_$TAG.csv"
  rm -rf "$TMPD"; echo
done

echo "ALL DONE $(date)"
