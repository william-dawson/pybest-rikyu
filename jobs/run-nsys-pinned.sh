#!/bin/bash
# Nsight Systems profile of PyTorch at 240 AO, BASELINE vs PINNED.
#
# The first trace (job 161307) profiled the unpatched code and found PyTorch's
# D2H costing 21.46 s against CuPy's 0.52 s for the same 65.8 GB. The pinning
# patch then cut CCSD at 240 AO by 37.3%, and PyBEST's own timer puts the whole
# saving in GPU: Generic, 67.2 -> 28.2 s. But that timer cannot separate kernels
# from allocation and transfer -- which is exactly what the first trace
# established -- so what the remaining 28.2 s consists of is currently inferred
# by subtracting one job's numbers from another's.
#
# Carrying the baseline trace across: kernels 3.41 s, H2D 2.23 s, and D2H
# nominally ~0.5 s once pinned, leaves roughly 22 s of GPU: Generic
# unattributed -- MORE than the fix recovered. This measures it directly:
#   1. does D2H actually fall to CuPy-like bandwidth, or only partway?
#   2. do the 1211 H2D calls cost host time beyond their 2.2 s of DMA?
#   3. where do the gaps sit -- cudaMalloc/cudaFree churn, or CPU work?
# Question 2 is the gate on whether caching Cholesky vectors in VRAM is worth
# any memory at all; on transfer time alone it is only 2% and not worth it.
#
# CuPy is not profiled: bounded pinning REGRESSED it by 7.5% at this size
# (job 161877), so the patch is PyTorch-only and CuPy's trace already exists.
#
# nsys is NOT in our container (cupy-cuda13x[ctk] ships libraries, not developer
# tools); RIKYU's `module load nvhpc` provides it.
#
# CRITICAL: nsys must run INSIDE the container. Invoking
# `nsys profile apptainer exec ... python` traces only the host process -- CUDA
# tracing works by injecting a preload library, and the container's mount
# namespace cannot see it, so the trace comes back with "does not contain CUDA
# kernel data" while still looking superficially valid.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
S=$P/systems
L=$P/logs; mkdir -p "$L"
W=/tmp/nsys-${SLURM_JOB_ID:-$$}; mkdir -p "$W"
export TMPDIR=$W

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} date=$(date)"

# ---------- discover nsys ----------
NSYS=""
if command -v nsys >/dev/null 2>&1; then NSYS=$(command -v nsys); fi
if [ -z "$NSYS" ]; then
  for m in nvhpc nvhpc-hpcx cuda nsight-systems; do
    module load "$m" >/dev/null 2>&1 && command -v nsys >/dev/null 2>&1 \
      && { NSYS=$(command -v nsys); echo "found via module $m"; break; }
    module unload "$m" >/dev/null 2>&1 || true
  done
fi
if [ -z "$NSYS" ]; then
  for c in /shared/software/*/bin/nsys /opt/nvidia/*/bin/nsys \
           /usr/local/cuda*/bin/nsys /shared/software/nvhpc*/Linux_aarch64/*/profilers/*/bin/nsys; do
    [ -x "$c" ] && { NSYS=$c; break; }
  done
fi
if [ -z "$NSYS" ]; then
  echo "NSYS_NOT_FOUND -- searched PATH, modules (nvhpc, cuda, nsight-systems) and"
  echo "common install prefixes. Listing what modules exist so the next attempt"
  echo "can be targeted:"
  module -t avail 2>&1 | grep -iE "nsight|nvhpc|cuda|profil" | head -20 || true
  echo "PREFLIGHT DONE $(date)"
  rm -rf "$W"; exit 0
fi
echo "NSYS=$NSYS"; "$NSYS" --version 2>&1 | head -2
# Bind the whole SDK tree: nsys needs its own libraries and helper binaries.
SDK=$(echo "$NSYS" | sed 's#/Linux_aarch64/.*##')
[ -d "$SDK" ] || SDK=$(dirname "$(dirname "$NSYS")")
echo "SDK=$SDK (bound into the container)"

# ---------- profile ----------
for STRAT in baseline pinned; do
  ENVV=PYBEST_PYTORCH_AVAIL=1
  TAG="nsyspin_h2o10_ccpvdz_pytorch_${STRAT}"
  echo "=============== $TAG ==============="

  # nsys INSIDE the container, with the SDK bound in so it is reachable there.
  # --trace: cuda for kernels+copies, osrt for CPU blocking calls (the gaps).
  $A exec --nv -B /data1 -B "$SDK" -B /tmp \
     --env ${ENVV},PYBEST_C_SPLITTING=1,PYBEST_TEMP=$W/pb-$STRAT,TMPDIR=$W,PYBEST_PIN_STRATEGY=$STRAT \
     "$SIF" "$NSYS" profile \
        --trace=cuda,osrt,nvtx \
        --cuda-memory-usage=true \
        --force-overwrite=true \
        --output="$W/$TAG" \
        python $S/h2o10_cc-pvdz_ccsd.py \
     > "$L/raw_$TAG.log" 2>&1
  rc=$?
  echo "profile rc=$rc"
  grep -aE "^# |Total energy" "$L/raw_$TAG.log" || true
  [ $rc -ne 0 ] && tail -15 "$L/raw_$TAG.log"

  REP="$W/$TAG.nsys-rep"
  if [ -f "$REP" ]; then
    echo "--- trace size: $(du -h "$REP" | cut -f1) ---"
    for R in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum cuda_api_sum; do
      echo "--- $R ---"
      $A exec -B /data1 -B "$SDK" -B /tmp "$SIF" \
        "$NSYS" stats --force-export=true --report "$R" --format table "$REP" 2>&1 \
        | grep -vE "^Processing|^Generating|^SQLite|^ *$" | head -22
    done
    # Keep the report: re-analysing later is far cheaper than re-running.
    cp "$REP" "$L/$TAG.nsys-rep" 2>/dev/null \
      && echo "kept $L/$TAG.nsys-rep ($(du -h "$L/$TAG.nsys-rep" | cut -f1))"
  else
    echo "no .nsys-rep produced"
  fi
  echo
done

rm -rf "$W"
echo "ALL DONE $(date)"
