#!/bin/bash
# Nsight Systems profile of a small molecular CCSD, both backends.
#
# Three questions the coarse counters cannot answer:
#   1. Where does the non-GEMM time go -- kernels, copies, or CPU gaps?
#   2. Why is CuPy's "GPU: Generic" path 2.3x slower than PyTorch's?
#   3. Why is CuPy's SCF ~10x slower for identical work?
#
# 240 AO is deliberate: ~105 s for CuPy, yet it contains SCF, the generic path
# AND C-split, so one short trace addresses all three. At 1150 AO the trace
# would be tens of GB. Not ncu -- it serialises kernels and destroys the overlap
# we are trying to measure.
#
# nsys is NOT in our container (cupy-cuda13x[ctk] ships libraries, not developer
# tools); RIKYU's `module load nvhpc` provides it (HPC SDK 26.3, Nsight Systems
# 2026.1.1).
#
# CRITICAL: nsys must run INSIDE the container. Invoking
# `nsys profile apptainer exec ... python` traces only the host process -- CUDA
# tracing works by injecting a preload library, and the container's mount
# namespace cannot see it, so the trace comes back with "does not contain CUDA
# kernel data" while still looking superficially valid (26 MB of osrt events).
# Instead bind the SDK into the container and run nsys from in there.
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
for BACKEND in cupy pytorch; do
  if [ "$BACKEND" = cupy ]; then ENVV=PYBEST_CUPY_AVAIL=1
  else ENVV=PYBEST_PYTORCH_AVAIL=1; fi
  TAG="nsys_h2o10_ccpvdz_${BACKEND}"
  echo "=============== $TAG ==============="

  # nsys INSIDE the container, with the SDK bound in so it is reachable there.
  # --trace: cuda for kernels+copies, osrt for CPU blocking calls (the gaps).
  $A exec --nv -B /data1 -B "$SDK" -B /tmp \
     --env ${ENVV},PYBEST_C_SPLITTING=1,PYBEST_TEMP=$W/pb-$BACKEND,TMPDIR=$W \
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
