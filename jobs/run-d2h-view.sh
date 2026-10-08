#!/bin/bash
# Is the staging copy worth removing where it is safe to remove?
#
# Our pinned patch copies out of the reused buffer on every transfer, because
# some callers retain the array. Job 161947 priced that copy -- 10.8 GB/s staged
# against 193 GB/s for the same DMA with no copy -- and job 162176 showed the
# retaining callers carry almost nothing:
#
#   td_GPU_helper:1487   95.6% of D2H bytes at 240 AO, 96.0% at 580   dest[...] +=
#   c_splitting:998       4.0% at 580 AO                              dest[...] +=
#   c_splitting:925       4.4% at 240 AO, 0.0% at 580                 RETURNED
#   c_splitting:1158      0.0%, 61 calls of zero bytes                RETURNED
#
# So: return a view where the caller consumes it inside the receiving expression,
# copy only where it escapes. Unrecognised callers copy, so the failure mode is
# correctness rather than corruption.
#
# Expected size: at 580 AO, 487 GB of D2H at 10.8 vs 193 GB/s is ~45 s against
# ~2.5 s, so about 42 s of an 810 s CCSD -- a few percent, consistent across
# sizes. The point is less the percent than that it demonstrates the upstream
# change: PyBEST already holds the destination at line 1487, so the DMA could
# write it directly and no staging buffer would exist at all.
#
# SYSLIST overrides the systems, so the same script covers 920 AO where a cell is
# ~70 min and the 240 AO pair would just be noise.
#
# Both cells also run the unravel fix, so this measures the third patch on top of
# the first two -- the configuration we would recommend. Paired in-job.
#
# The ENERGY CHECK is the real test here: a view returned to a caller that
# retained it would corrupt results silently, so an energy that still matches
# -763.24436502 (580 AO) and -762.39447031 (240 AO) is what licenses the patch.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
S=$P/systems
L=$P/logs; mkdir -p "$L"
W=/tmp/d2hv-${SLURM_JOB_ID:-$$}

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
echo

for SYS in ${SYSLIST:-h2o10_cc-pvdz_ccsd h2o10_cc-pvtz_ccsd}; do
  for MODE in copy view; do
    if [ "$MODE" = copy ]; then EXTRA=PYBEST_PIN_STRATEGY=pinned
    else EXTRA=PYBEST_D2H_VIEW=1; fi
    TAG="d2hv_${SYS}_${MODE}"; RAW=$L/raw_$TAG.log
    TMPD=$W-$SYS-$MODE; mkdir -p "$TMPD"
    echo "=============== $TAG ==============="
    /usr/bin/time -v $A exec --nv -B /data1 -B /tmp \
       --env PYBEST_PYTORCH_AVAIL=1,PYBEST_C_SPLITTING=1,PYBEST_TEMP=$TMPD,PYBEST_UNRAVEL_FIX=1,PYBEST_BENCH_DIR=$P,${EXTRA} \
       "$SIF" python $S/${SYS}.py > "$RAW" 2>&1
    rc=$?
    grep -aE "^(# |Total energy|Correlation energy)" "$RAW"
    grep -aE "Maximum resident set size" "$RAW" || true
    # A patch that does not install is silent -- an env var read by a driver that
    # does not carry the block is a no-op, and the cell then measures something
    # else entirely. This happened once: the 920 AO driver had not been
    # re-uploaded after the view block was added, so the "view" cell ran
    # UNPINNED and the 65 minutes measured the wrong thing. Fail loudly instead.
    for want in "unravel: assign_triu" \
                "$([ "$MODE" = view ] && echo 'd2h_view: view where' || echo 'pin: pytorch transfers')"; do
      grep -aqF "$want" "$RAW" \
        || { echo "PATCH DID NOT INSTALL: expected \"$want\" in the log."; rc=99; }
    done
    sed -n '/Overview of CPU time usage/,/Page swaps/p' "$RAW" \
      | grep -aE "^(GPU: |Base: contract|RCCSD: unravel|Ints: CD-ERI|Total )" || true
    echo "rc=$rc"
    [ $rc -ne 0 ] && tail -12 "$RAW"
    rm -rf "$TMPD"; echo
  done
done

echo "ALL DONE $(date)"
