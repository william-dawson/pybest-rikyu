#!/bin/bash
# Does fixing host-memory pinning help? Small (240 AO) and medium (580 AO),
# both backends, baseline vs patched. 8 cells.
#
# Baseline numbers to beat, from nsys job 161307 at 240 AO:
#   CuPy     CC 87.0 s;  cudaHostAlloc 12.74 s + cudaFreeHost 2.60 s (1032 calls)
#   PyTorch  CC 115.7 s; D2H 21.46 s vs CuPy's 0.52 s for the same 65.8 GB
# and from the clean runs: CuPy 20.4 s/iter at 240, 4.12 m at 580;
#                          PyTorch 23.6 s/iter at 240, 3.61 m at 580.
# NOTE: deliberately NOT setting PYBEST_TEMP. pin_experiment.py imports pybest
# before running the driver, so the global FileManager is constructed with the
# default "pybest-temp"; the driver then repoints temp_dir and PyBEST ends up
# looking for checkpoint_cd-eri.h5 in the original directory. 240 and 580 AO do
# not need node-local scratch (both ran fine without it), so the simplest fix is
# to leave temp_dir alone and keep one directory in play.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
L=$P/logs; mkdir -p "$L"

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
nvidia-smi --query-gpu=index,name --format=csv,noheader
echo

for SYS in h2o10_cc-pvdz_ccsd h2o10_cc-pvtz_ccsd; do
  for BACKEND in cupy pytorch; do
    if [ "$BACKEND" = cupy ]; then ENVV=PYBEST_CUPY_AVAIL=1
    else ENVV=PYBEST_PYTORCH_AVAIL=1; fi
    for STRAT in baseline pinned; do
      TAG="pin_${SYS}_${BACKEND}_${STRAT}"
      RAW=$L/raw_$TAG.log
      echo "=============== $TAG ==============="
      $A exec --nv -B /data1 \
         --env ${ENVV},PYBEST_C_SPLITTING=1 \
         "$SIF" python $P/pin_experiment.py \
            --driver $P/systems/${SYS}.py --strategy $STRAT > "$RAW" 2>&1
      rc=$?
      grep -aE "^# |^Total energy|^Correlation energy" "$RAW"
      echo "--- timer: GPU and transfer-adjacent sections ---"
      sed -n '/Overview of CPU time usage/,/Page swaps/p' "$RAW" \
        | grep -aE "^(GPU: |Base: contract|RCCSD: unravel|RCCSD: VecFct|Total |SCF )" || true
      echo "rc=$rc"
      [ $rc -ne 0 ] && tail -20 "$RAW"
      echo
    done
  done
done
echo "ALL DONE $(date)"
