#!/bin/bash
# Does removing the `factor * X` temporary pay, and does it compose with
# dropping the allocator flush?
#
# cProfile at 920 AO put 287.1 s -- 16% of a 1827 s CCSD -- in the OWN time of
# base.py:contract (142.2 s / 473 calls) and td_GPU_helper (144.9 s / 243).
# At 300-600 ms per call that is not interpreter overhead; it is the numpy bulk
# arithmetic of `arr[slice_] += factor * X`, attributed to the enclosing frame
# because `+=` on a slice is bytecode rather than a profiled call. `factor * X`
# allocates a 9.02 GiB temporary at that size and first-touches every page of
# it, thirteen call sites over.
#
# Four cells, paired in one job, all on top of the three accepted patches:
#   base      the current best configuration
#   accum     + factor * X scaled in place, no temporary
#   noflush   + clean_memory() skipped (measured -7.4% on its own)
#   both      the two together
#
# `both` is the cell that matters: the flush and the temporary are plausibly
# the same cost seen twice, since a 9 GiB allocation after an allocator flush
# is exactly what makes the next cudaMalloc expensive. If they do not compose,
# that is the explanation.
#
# 580 AO at maxiter=4 so the figures are comparable with every earlier
# molecular measurement, ~10 min a cell.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
S=$P/systems
L=$P/logs; mkdir -p "$L"
W=/tmp/accum-${SLURM_JOB_ID:-$$}
SYS=${SYSNAME:-h2o10_cc-pvtz_ccsd}
BASE="PYBEST_PYTORCH_AVAIL=1,PYBEST_C_SPLITTING=1,PYBEST_D2H_VIEW=1,PYBEST_UNRAVEL_FIX=1,PYBEST_BENCH_DIR=$P"

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
echo "system=$SYS"; echo

for MODE in base accum noflush both; do
  case $MODE in
    base)    EXTRA="" ;;
    accum)   EXTRA=",PYBEST_ACCUM=1" ;;
    noflush) EXTRA=",PYBEST_NOFLUSH=1" ;;
    both)    EXTRA=",PYBEST_ACCUM=1,PYBEST_NOFLUSH=1" ;;
  esac
  TAG="accum_${SYS}_${MODE}"; RAW=$L/raw_$TAG.log
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
  # Every requested patch must announce itself; a silent no-op has cost an hour.
  for want in "d2h_view: view where" "unravel: assign_triu"; do
    grep -aqF "$want" "$RAW" || echo "PATCH DID NOT INSTALL: $want"
  done
  case $MODE in
    accum|both) grep -aqF "# accum:" "$RAW" || echo "PATCH DID NOT INSTALL: accum" ;;
  esac
  case $MODE in
    noflush|both) grep -aqF "# noflush:" "$RAW" || echo "PATCH DID NOT INSTALL: noflush" ;;
  esac
  echo "rc=$rc"; [ $rc -ne 0 ] && tail -12 "$RAW"
  rm -rf "$TMPD"; echo
done

echo "ALL DONE $(date)"
