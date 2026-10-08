#!/bin/bash
# Does the unravel fix hold up in a real CCSD, on top of pinning?
#
# The microbenchmark (job 161965) measured 13.7 s -> 2.5 s per unravel at 920 AO
# dimensions, bitwise identical at three sizes and two block sizes. PyBEST's
# timer charges 462 s to RCCSD: unravel at 920 AO and 144.6 s at 580 AO, both
# about 10% of the run, so a 5.4x there is worth ~8% of the whole calculation.
#
# One caveat the microbenchmark exposes and this job should settle: 462 s divided
# by 13.7 s is roughly 34 calls, but maxiter is 4. So unravel is being invoked
# about eight times per CC iteration -- worth knowing on its own, and it means
# the per-call speedup multiplies over more calls than expected, not fewer.
#
# Both cells are PINNED, so the unravel fix is measured on top of the transfer
# fix rather than against an unfixed baseline -- that is the configuration we
# would actually recommend. Paired in-job: cross-job drift on identical cells
# has reached 9%, and the effect sought here is ~8-10%.
#
# 580 AO runs first (~25 min a cell) so a result arrives before the 920 AO pair
# (~80 min a cell) finishes.
#
# PYBEST_TEMP is node-local NVMe and not optional: PyBEST writes a checkpoint and
# reads it back with no fsync and no retry, so on Lustre the read fails
# nondeterministically. PYBEST_BENCH_DIR is where unravel_patch.py lives.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
S=$P/systems
L=$P/logs; mkdir -p "$L"
W=/tmp/unrav-${SLURM_JOB_ID:-$$}

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
nvidia-smi --query-gpu=index,name --format=csv,noheader; df -h /tmp | tail -1
echo

for SYS in h2o10_cc-pvtz_ccsd h2o10_aug-cc-pvtz_ccsd; do
  for FIX in 0 1; do
    TAG="unrav_${SYS}_pinned_fix${FIX}"; RAW=$L/raw_$TAG.log
    TMPD=$W-$SYS-$FIX; mkdir -p "$TMPD"
    echo "=============== $TAG ==============="
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used \
               --format=csv,noheader -l 30 > "$L/gpu_$TAG.csv" 2>/dev/null &
    SMI=$!
    /usr/bin/time -v $A exec --nv -B /data1 -B /tmp \
       --env PYBEST_PYTORCH_AVAIL=1,PYBEST_C_SPLITTING=1,PYBEST_TEMP=$TMPD,PYBEST_PIN_STRATEGY=pinned,PYBEST_UNRAVEL_FIX=$FIX,PYBEST_BENCH_DIR=$P \
       "$SIF" python $S/${SYS}.py > "$RAW" 2>&1
    rc=$?
    kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
    grep -aE "^(# |Total energy|Correlation energy)" "$RAW"
    # Peak RSS: the fix should REDUCE it by removing ~27 GiB of temporaries.
    grep -aE "Maximum resident set size" "$RAW" || true
    sed -n '/Overview of CPU time usage/,/Page swaps/p' "$RAW" \
      | grep -aE "^(GPU: |Base: contract|RCCSD: unravel|RCCSD: VecFct|Ints: CD-ERI|SCF |Total )" || true
    echo "rc=$rc"
    [ $rc -ne 0 ] && tail -15 "$RAW"
    awk -F, '{gsub(/ |%/,"",$2); if ($2+0>g) g=$2+0} END{print "gpu_max_util="g"%"}' \
        "$L/gpu_$TAG.csv"
    rm -rf "$TMPD"; echo
  done
done

echo "ALL DONE $(date)"
