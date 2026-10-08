#!/bin/bash
# Nsight Compute Speed Of Light on the ladder GEMM, testing one hypothesis.
#
# The same term 14.4 runs at ~74% of FP64 peak in the synthetic benchmark
# (o=100) and ~14% in molecular CCSD (o=40). The proposed reason: the second
# step reshapes to M=v^2, N=o^2, K=v^2, so o^2 is the SHORT dimension and
# frozen-core molecular runs have small o.
#
# This varies ONLY o, holding v fixed at 300, so nothing else can explain a
# difference:
#     N=400 nocc=100 -> v=300, N=o^2=10000
#     N=340 nocc=40  -> v=300, N=o^2= 1600
#
# SpeedOfLight reports Compute (SM) and Memory throughput each as % of peak and
# names the bound. If the o=40 case shows LOW on both, it is latency-bound and
# the skinny-GEMM explanation holds. If it is memory-bound instead, the
# explanation is wrong.
#
# ncu replays each kernel to collect counters, so --launch-count keeps this cheap
# and --kernel-name skips the copies. Same in-container requirement as nsys.
#
# --kernel-name matches the FUNCTION name. The CUTLASS GEMMs demangle to
#   void cutlass::Kernel2<cutlass_80_tensorop_d884gemm_...>(T1::Params)
# so the function is `Kernel2` and "d884gemm" lives only in the template
# parameter: matching on it profiles nothing and ncu reports
#   ==WARNING== No kernels were profiled.  Available Kernels: 1. Kernel2
# Only two kernels exist in these runs, so `Kernel2` selects exactly the GEMMs.
#
# And do NOT use --launch-skip here. It skips the first N MATCHING launches, and
# a single ladder pass at N=400 has only a handful of GEMM launches, so
# `--launch-skip 4` left nothing to profile: ncu reported "No kernels were
# profiled" while listing only cupy_copy as unprofiled, i.e. Kernel2 matched but
# was skipped past. Profile the first launches instead.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
L=$P/logs; mkdir -p "$L"
W=/tmp/ncu-${SLURM_JOB_ID:-$$}; mkdir -p "$W"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} date=$(date)"

NCU=""
if command -v ncu >/dev/null 2>&1; then NCU=$(command -v ncu); fi
if [ -z "$NCU" ]; then
  module load nvhpc >/dev/null 2>&1 && command -v ncu >/dev/null 2>&1 && NCU=$(command -v ncu)
fi
[ -z "$NCU" ] && for c in /shared/software/hpc_sdk/Linux_aarch64/*/compilers/bin/ncu \
                          /shared/software/hpc_sdk/Linux_aarch64/*/profilers/*/ncu; do
  [ -x "$c" ] && { NCU=$c; break; }
done
if [ -z "$NCU" ]; then
  echo "NCU_NOT_FOUND"; module -t avail 2>&1 | grep -iE "nsight|nvhpc" | head; exit 0
fi
echo "NCU=$NCU"; "$NCU" --version 2>&1 | head -3
SDK=/shared/software/hpc_sdk
echo

# label:N:nocc  -- v = N - nocc = 300 in both
for SPEC in wide:400:100 skinny:340:40; do
  LBL=${SPEC%%:*}; rest=${SPEC#*:}; N=${rest%%:*}; OCC=${rest##*:}
  for BACKEND in cupy pytorch; do
    if [ "$BACKEND" = cupy ]; then ENVV=PYBEST_CUPY_AVAIL=1
    else ENVV=PYBEST_PYTORCH_AVAIL=1; fi
    TAG="ncu_${LBL}_N${N}_o${OCC}_${BACKEND}"
    echo "=============== $TAG  (v=$((N-OCC)), N_gemm=o^2=$((OCC*OCC))) ==============="

    $A exec --nv -B /data1 -B "$SDK" -B /tmp \
       --env ${ENVV},PYBEST_C_SPLITTING=1,TMPDIR=$W \
       "$SIF" "$NCU" \
          --target-processes all \
          --kernel-name regex:Kernel2 \
          --launch-count 4 \
          --section SpeedOfLight --section LaunchStats --section Occupancy \
          python $P/repro_table4.py --nbasis "$N" --nocc "$OCC" \
             --nvec-factor 5 --reps 1 --warmup 0 \
       > "$L/raw_$TAG.log" 2>&1
    rc=$?
    echo "rc=$rc"
    if grep -qaE "ERR_NVGPUCTRPERM|insufficient permission" "$L/raw_$TAG.log"; then
      echo "PERMISSION DENIED for GPU counters -- needs the admin to allow"
      echo "non-root profiling, or run with elevated privileges."
      grep -am2 -E "ERR_NVGPUCTRPERM|insufficient permission" "$L/raw_$TAG.log"
    fi
    # The three lines that answer the question, plus the shape.
    grep -aE "Compute \(SM\) Throughput|Memory Throughput|DRAM Throughput|Duration|Grid Size|Block Size|Achieved Occupancy|^ *(d884|void)" \
      "$L/raw_$TAG.log" | head -40
    echo "--- bound statement ---"
    grep -aA3 -E "OPT|INF|WRN" "$L/raw_$TAG.log" | head -20 || true
    [ $rc -ne 0 ] && tail -15 "$L/raw_$TAG.log"
    echo
  done
done
rm -rf "$W"
echo "ALL DONE $(date)"
