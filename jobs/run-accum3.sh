#!/bin/bash
# The accumulate through PyTorch, and this time prove it is correct.
#
# Previous attempts:
#   accum   scaled X in place        +0.5%   same traffic, so no change
#   accum2  chunked over threads    -12.5%   WRONG: energy moved 3.9e-4 Ha
#
# accum2's bug was in its fallback branch, which scaled every operand that was
# not literally inputs[0] instead of only the operand that was self, so
# np.add(self, other) scaled `other` too. 150 of 424 accumulates took it.
#
# accum3 uses torch.from_numpy(dest).add_(torch.from_numpy(src), alpha=factor):
# a genuine fused axpy, 2.36 TB instead of 3.94, threaded by ATen, zero-copy
# both ways. The fallback now materialises SELF already scaled and reruns the
# ufunc with every other operand untouched, so there is no operand rewriting
# left to get wrong. Verified locally against the exact idiom at seven shapes
# and six factors, the np.add(self, other) case that broke accum2, three other
# ufuncs, non-contiguous and sliced destinations, views of a pending-scaled
# array, and repeated use.
#
# THE ENERGY IS THE RESULT. -763.24436502 at 580 AO, maxiter=4. A faster wrong
# number is worth nothing, and the last round produced exactly that.
#
# Cells:
#   base      control
#   accum3    with PYBEST_ACCUM3_VERIFY=20, which checks the first twenty fused
#             operations against a numpy reference and reports the largest
#             deviation -- runtime evidence, not just unit tests
#   accum3_t16  ATen thread count pinned, since accum2 showed almost no thread
#             sensitivity and that deserves a second look
#   both      accum3 + noflush, the two surviving wins together
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
S=$P/systems
L=$P/logs; mkdir -p "$L"
W=/tmp/accum3-${SLURM_JOB_ID:-$$}
SYS=${SYSNAME:-h2o10_cc-pvtz_ccsd}
BASE="PYBEST_PYTORCH_AVAIL=1,PYBEST_C_SPLITTING=1,PYBEST_D2H_VIEW=1,PYBEST_UNRAVEL_FIX=1,PYBEST_BENCH_DIR=$P"
REF="-763.24436502"

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
echo "system=$SYS  reference energy=$REF"; echo

for MODE in base accum3 accum3_t16 both; do
  case $MODE in
    base)       EXTRA="" ;;
    accum3)     EXTRA=",PYBEST_ACCUM3=1,PYBEST_ACCUM3_VERIFY=20" ;;
    accum3_t16) EXTRA=",PYBEST_ACCUM3=1,PYBEST_ACCUM3_THREADS=16" ;;
    both)       EXTRA=",PYBEST_ACCUM3=1,PYBEST_NOFLUSH=1" ;;
  esac
  TAG="accum3_${SYS}_${MODE}"; RAW=$L/raw_$TAG.log
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
    accum3*|both) grep -aqF "# accum3:" "$RAW" || echo "PATCH DID NOT INSTALL: accum3" ;;
  esac
  # The whole point. Say so unmissably either way.
  # -- and -F are both required: the reference starts with a minus, which grep
  # otherwise parses as an option bundle, so the test never matches and every
  # cell reports WRONG. That is a false alarm on a correct result, which is a
  # worse failure than no check at all.
  if grep -aqF -- "$REF" "$RAW"; then
    echo "ENERGY OK ($REF)"
  else
    echo "!!! ENERGY WRONG -- this cell's timing is meaningless"
    grep -a "Total energy" "$RAW" | tail -1
  fi
  echo "rc=$rc"; [ $rc -ne 0 ] && tail -12 "$RAW"
  rm -rf "$TMPD"; echo
done

echo "ALL DONE $(date)"
