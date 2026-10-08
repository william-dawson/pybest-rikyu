#!/bin/bash
# Run one generated PyBEST benchmark under both GPU backends, C-split.
#   bash run-ccsd.sh <system-label>        e.g. h2o10_cc-pvtz_ccsd
#
# Published per-CCSD-iteration references (Dobrowolska et al., Table 5):
#   h2o10_cc-pvdz (240 AO)  GH200 CuPy 23.9 s   PyTorch 25.4 s
#   h2o10_cc-pvtz (580 AO)  GH200 CuPy  5.5 m   PyTorch  5.7 m
#
# The deliverable is the per-section timer table, not wall clock: it separates
# the CPU-only setup (libint, libchol) from the offloaded contractions, and
# splits GPU time into the optimised C-split path and the generic one.
set -uo pipefail
SYS=${1:?usage: run-ccsd.sh <system-label>}
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
S=$P/systems
L=$P/logs; mkdir -p "$L"
# Optional 2nd arg: which backends to run (default both), e.g. "pytorch".
BACKENDS=${2:-"cupy pytorch"}

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} sys=$SYS date=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
free -g | sed -n 2p
df -h /tmp | tail -1
echo

for BACKEND in $BACKENDS; do
  if [ "$BACKEND" = cupy ]; then ENVV=PYBEST_CUPY_AVAIL=1
  else ENVV=PYBEST_PYTORCH_AVAIL=1; fi
  TAG="${SYS}_${BACKEND}_Csplit"
  RAW=$L/raw_$TAG.log
  echo "=============== $TAG ==============="

  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw \
             --format=csv,noheader -l 15 > "$L/gpu_$TAG.csv" 2>/dev/null &
  SMI=$!

  # Checkpoint spills go to node-local NVMe, not Lustre. Per backend so the two
  # cells cannot share a directory, and under the job id so nothing outlives it.
  TMPD=/tmp/pybest-${SLURM_JOB_ID:-$$}-$BACKEND
  mkdir -p "$TMPD"

  $A exec --nv -B /data1 \
     --env ${ENVV},PYBEST_C_SPLITTING=1,PYBEST_TEMP=$TMPD \
     "$SIF" python $S/${SYS}.py > "$RAW" 2>&1
  rc=$?
  echo "tmpdir_peak=$(du -sh "$TMPD" 2>/dev/null | cut -f1) at $TMPD"
  rm -rf "$TMPD"

  kill $SMI 2>/dev/null; wait $SMI 2>/dev/null

  grep -aE "^# |^Total energy|^Correlation energy|^T1 diagnostic" "$RAW"
  echo "--- warnings ---"
  grep -aiE "warning|fallback|MemoryError" "$RAW" | sort -u | head -12
  echo "--- timer table ---"
  sed -n '/Overview of CPU time usage/,/Page swaps/p' "$RAW" | tail -35
  echo "rc=$rc  raw=$RAW"
  awk -F, '{gsub(/ |%/,"",$2); if ($2+0>m) m=$2+0} END{print "gpu_max_util="m"%"}' \
      "$L/gpu_$TAG.csv"
  echo
done
echo "ALL DONE $(date)"
