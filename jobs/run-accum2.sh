#!/bin/bash
# Fused, threaded accumulate: is the 287 s really just one core?
#
# `arr[slice_] += factor * X` is 287 s at 920 AO, 16% of the CCSD phase and the
# largest host-side item. The first patch removed the temporary and changed
# nothing (+0.5%, job 165252) -- correctly, because the traffic is the same
# either way:
#     today        tmp = factor*X ; dest += tmp    3.94 TB
#     accum        X *= factor    ; dest += X      3.94 TB
#     fused        dest += factor*X, one pass      2.36 TB
# The rate is the real story: 3.94 TB in 287 s is 13.7 GB/s, which is 4% of ONE
# Grace socket. numpy's += is single-threaded.
#
# accum2 records the factor instead of applying it, intercepts the += through
# __array_ufunc__, and does a chunked threaded fused accumulate. Chunks stay in
# cache so the scaling is free. Predicted 287 s -> 12-25 s; locally it is 2.2x
# on ten laptop cores, where bandwidth runs out immediately.
#
# Cells, all on the three accepted patches, 580 AO maxiter=4:
#   base     control
#   accum2   fused + threaded
#   t8/t64   thread-count sensitivity, to show it is bandwidth and not luck
#   both     accum2 + noflush, since both target host-side stalls
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
S=$P/systems
L=$P/logs; mkdir -p "$L"
W=/tmp/accum2-${SLURM_JOB_ID:-$$}
SYS=${SYSNAME:-h2o10_cc-pvtz_ccsd}
BASE="PYBEST_PYTORCH_AVAIL=1,PYBEST_C_SPLITTING=1,PYBEST_D2H_VIEW=1,PYBEST_UNRAVEL_FIX=1,PYBEST_BENCH_DIR=$P"

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
echo "system=$SYS"; echo

for MODE in base accum2 accum2_t8 accum2_t64 both; do
  case $MODE in
    base)        EXTRA="" ;;
    accum2)      EXTRA=",PYBEST_ACCUM2=1" ;;
    accum2_t8)   EXTRA=",PYBEST_ACCUM2=1,PYBEST_ACCUM2_THREADS=8" ;;
    accum2_t64)  EXTRA=",PYBEST_ACCUM2=1,PYBEST_ACCUM2_THREADS=64" ;;
    both)        EXTRA=",PYBEST_ACCUM2=1,PYBEST_NOFLUSH=1" ;;
  esac
  TAG="accum2_${SYS}_${MODE}"; RAW=$L/raw_$TAG.log
  TMPD=$W-$MODE; mkdir -p "$TMPD"
  echo "=============== $TAG ==============="
  /usr/bin/time -v $A exec --nv -B /data1 -B /tmp \
     --env ${BASE},PYBEST_TEMP=$TMPD${EXTRA} \
     "$SIF" python $S/${SYS}.py > "$RAW" 2>&1
  rc=$?
  grep -aE "^(# |Total energy)" "$RAW"
  grep -aE "Maximum resident set size" "$RAW" || true
  sed -n '/Overview of CPU time usage/,/Page swaps/p' "$RAW" \
    | grep -aE "^(GPU: |Base: contract|RCCSD: unravel|Ints: CD-ERI|Total )" || true
  for want in "d2h_view: view where" "unravel: assign_triu"; do
    grep -aqF "$want" "$RAW" || echo "PATCH DID NOT INSTALL: $want"
  done
  case $MODE in
    accum2*|both) grep -aqF "# accum2:" "$RAW" || echo "PATCH DID NOT INSTALL: accum2" ;;
  esac
  [ "$MODE" = both ] && { grep -aqF "# noflush:" "$RAW" || echo "PATCH DID NOT INSTALL: noflush"; }
  # A high fallback count would mean the destinations are not contiguous and the
  # fused path is rarely taken -- that would explain a null result.
  grep -aE "^# accum2: fused" "$RAW" || true
  echo "rc=$rc"; [ $rc -ne 0 ] && tail -12 "$RAW"
  rm -rf "$TMPD"; echo
done

echo "ALL DONE $(date)"
