#!/bin/bash
# Validate the setup cache: miss, then hit, then check nothing moved.
#
# The Cholesky decomposition is 123 s at 580 AO and 628-1027 s at 920, does not
# depend on anything we vary between cells, and VARIES between jobs by up to
# 1.8x for identical work -- which is a large part of why cross-job totals have
# not been comparable. Caching it removes the cost and the variance.
#
# The orbitals are cached too but are NOT used to skip SCF: they are passed to
# RHF as its initial guess, so it converges in an iteration or two through the
# normal path.
#
# Three cells at 240 AO, ~2 min each:
#   cold   no cache directory at all, the reference
#   miss   cache directory empty: computes, then writes
#   hit    cache present: loads
# The energies must agree to every digit across all three. If the cached
# orbitals were wrong, SCF would simply converge away from them and the timing
# would show it, so both failure modes are visible.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
S=$P/systems
L=$P/logs; mkdir -p "$L"
C=$P/setupcache; rm -rf "$C"; mkdir -p "$C"
W=/tmp/ctest-${SLURM_JOB_ID:-$$}
SYS=${SYSNAME:-h2o10_cc-pvdz_ccsd}

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
echo "system=$SYS cache=$C"; echo

for MODE in cold miss hit; do
  if [ "$MODE" = cold ]; then EXTRA=""; else EXTRA=",PYBEST_SETUP_CACHE=$C"; fi
  TAG="ctest_${SYS}_${MODE}"; RAW=$L/raw_$TAG.log
  TMPD=$W-$MODE; mkdir -p "$TMPD"
  echo "=============== $TAG ==============="
  $A exec --nv -B /data1 -B /tmp \
     --env PYBEST_PYTORCH_AVAIL=1,PYBEST_C_SPLITTING=1,PYBEST_TEMP=$TMPD,PYBEST_MAXITER=2${EXTRA} \
     "$SIF" python $S/${SYS}.py > "$RAW" 2>&1
  rc=$?
  grep -aE "^# (setup_cache|cholesky_eri_sec|rhf_sec|ccsd_total_sec|total_sec)|Total energy" "$RAW"
  echo "rc=$rc"; [ $rc -ne 0 ] && tail -12 "$RAW"
  rm -rf "$TMPD"; echo
done

echo "--- cache contents ---"
ls -la "$C" 2>/dev/null
echo
echo "--- energies must be identical across all three ---"
grep -ah "Total energy" $L/raw_ctest_${SYS}_*.log | sort -u
echo "(one line above = all three agree)"
echo "ALL DONE $(date)"
