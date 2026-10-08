#!/bin/bash
# Does the CPU side of CCSD scale with threads?
#
# At 1150 AO: 120,844 s CPU time against 13,007 s wall = 9.3x on 64 cores,
# about 15% efficiency. At 240 AO: 760 s against 105 s = 7.2x. OMP_NUM_THREADS
# has never been set in any run so far.
#
# But that average is suspect, and the point of this job is to decompose it.
# The Cholesky decomposition (libchol, OpenMP) is genuinely threaded and could
# account for most of the CPU time on its own -- at 240 AO, 5.7 s of wall time
# on 64 threads would be ~365 s of the 760 s CPU total. Meanwhile the CCSD-side
# CPU work is numpy reshape and indexing (RCCSD: unravel) plus Python dispatch
# (Base: contract own), which are single-threaded regardless of OMP settings.
#
# So read the TIMER SECTIONS, not wall clock:
#   Ints: CD-ERI        expected to scale    (setup, not our target)
#   RCCSD: unravel      expected flat        (numpy indexing, serial)
#   Base: contract own  expected flat        (Python dispatch)
#   GPU: *              expected flat        (host-side plumbing, serial)
#
# A negative result is the useful one: if the CCSD sections are flat, then
# throwing cores at the host-side 30+ s will not help and the allocation and
# transfer churn is the only lever left.
#
# 580 AO, PyTorch (faster at this size, and its baseline is known good).
# Driver run DIRECTLY -- no pin_experiment.py wrapper, which breaks PyBEST's
# cache dump/load path whenever nact > 300.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
L=$P/logs; mkdir -p "$L"

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
echo

for T in 64 32 16 8; do
  TAG="threads_${T}_h2o10_cc-pvtz_pytorch"
  RAW=$L/raw_$TAG.log
  echo "=============== OMP_NUM_THREADS=$T ==============="
  /usr/bin/time -f "wall=%e s  cpu_user=%U s  cpu_sys=%S s  maxrss=%M kB" \
  $A exec --nv -B /data1 \
     --env PYBEST_PYTORCH_AVAIL=1,PYBEST_C_SPLITTING=1,OMP_NUM_THREADS=$T,OPENBLAS_NUM_THREADS=$T \
     "$SIF" python $P/systems/h2o10_cc-pvtz_ccsd.py > "$RAW" 2>&1
  rc=$?
  grep -aE "^# |^Total energy" "$RAW"
  echo "--- sections that should scale vs should not ---"
  sed -n '/Overview of CPU time usage/,/Page swaps/p' "$RAW" \
    | grep -aE "^(Ints: CD-ERI|RCCSD: unravel|Base: contract|GPU: C-split|GPU: Generic|RCCSD: VecFct|SCF |Total )" || true
  grep -aE "^(CPU user time|CPU sysem time)" "$RAW" || true
  grep -aE "wall=" "$RAW" || true
  echo "rc=$rc"
  [ $rc -ne 0 ] && tail -15 "$RAW"
  echo
done
echo "ALL DONE $(date)"
