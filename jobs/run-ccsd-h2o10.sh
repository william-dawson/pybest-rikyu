#!/bin/bash
# Vanilla CCSD on (H2O)10 / cc-pVDZ (240 AO) -- the molecular pipeline check.
#
# This is Table 5's first block. Published per-CCSD-iteration times:
#   GH200 CuPy 23.9 s | GH200 PyTorch 25.4 s | Grace CPU 72c 52 s
#
# Purpose is correctness first: geometry -> SCF -> Cholesky ERI -> CCSD, with
# log.level=high so a silent CPU fallback shows up as a warning rather than as
# a plausible-looking number.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
S=$P/systems
R=$P/results; mkdir -p "$R"
L=$P/logs;   mkdir -p "$L"

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo

for BACKEND in cupy pytorch; do
  if [ "$BACKEND" = cupy ]; then ENVV=PYBEST_CUPY_AVAIL=1
  else ENVV=PYBEST_PYTORCH_AVAIL=1; fi
  TAG="h2o10_ccpvdz_ccsd_${BACKEND}_Csplit"
  RAW=$L/raw_$TAG.log
  echo "=============== $TAG ==============="

  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw \
             --format=csv,noheader -l 5 > "$L/gpu_$TAG.csv" 2>/dev/null &
  SMI=$!

  $A exec --nv -B /data1 \
     --env ${ENVV},PYBEST_C_SPLITTING=1 \
     "$SIF" python $S/h2o10_cc-pvdz_ccsd.py > "$RAW" 2>&1
  rc=$?

  kill $SMI 2>/dev/null; wait $SMI 2>/dev/null

  # Our own markers, PyBEST's per-iteration table, and any fallback warning.
  grep -aE "^# |^ *Iter|^ *[0-9]+ +-?[0-9]" "$RAW" | head -30
  echo "--- warnings ---"
  grep -aiE "warning|fallback|not enough memory|MemoryError|cpu" "$RAW" | head -15
  echo "--- timer table ---"
  sed -n '/Overview of CPU time usage/,/^ *CPU user time/p' "$RAW" | head -30
  echo "rc=$rc  raw=$RAW"
  awk -F, '{gsub(/ |%/,"",$2); if ($2+0>m) m=$2+0} END{print "gpu_max_util="m"%"}' \
      "$L/gpu_$TAG.csv"
  echo
done
echo "ALL DONE $(date)"
