#!/bin/bash
# Categorise the time at a size where C-split dominates. Everything we know at
# the CUDA level comes from 240 AO, where GPU: C-split is 3.5 s of 70 s of GPU
# time. At 920 AO it is 1782 s of 2946 -- so we have no kernel-level data at all
# from the regime that decides large-system performance.
#
# Two facts cannot both be acted on until this runs:
#   * the FLOP model puts integral reconstruction at 78.5% of CC arithmetic
#   * the ladder reaches 14% of DGEMM peak while its kernels sit at 98% of
#     roofline (Nsight Compute)
# Together those say ~86% of C-split's time is not arithmetic. This job finds
# out what it is, in two complementary ways.
#
#   cell 1  cProfile  -- attributes HOST time to PyBEST functions. The nsys
#                        trace at 240 AO left ~78% of the run as "host-side
#                        Python/numpy" with no further breakdown; this is that
#                        breakdown, and it needs no GPU tooling.
#   cell 2  nsys      -- kernels, copies, CUDA API and OS-runtime blocking calls
#                        at this size, to compare against the 240 AO trace.
#
# Both run the BEST configuration (view + unravel), because what matters is what
# remains after the fixes, not what the unpatched code did.
#
# PYBEST_MAXITER=2 keeps the nsys trace analysable: 4 iterations at ~740 s each
# would give roughly a gigabyte. Two iterations still show the per-iteration
# structure, and the setup phase is identical either way.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
S=$P/systems
L=$P/logs; mkdir -p "$L"
R=$P/results/profile920; mkdir -p "$R"
W=/tmp/prof920-${SLURM_JOB_ID:-$$}; mkdir -p "$W"
export TMPDIR=$W
SYS=h2o10_aug-cc-pvtz_ccsd
COMMON="PYBEST_PYTORCH_AVAIL=1,PYBEST_C_SPLITTING=1,PYBEST_D2H_VIEW=1,PYBEST_UNRAVEL_FIX=1,PYBEST_BENCH_DIR=$P,PYBEST_MAXITER=2"

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
nvidia-smi --query-gpu=index,name --format=csv,noheader; df -h /tmp | tail -1
echo

# ---------------- cell 1: host-side attribution ----------------
TAG=prof920_cprofile; RAW=$L/raw_$TAG.log
echo "=============== $TAG ==============="
mkdir -p "$W/pb-cp"
$A exec --nv -B /data1 -B /tmp \
   --env ${COMMON},PYBEST_TEMP=$W/pb-cp \
   "$SIF" python -m cProfile -o $W/$TAG.prof $S/${SYS}.py > "$RAW" 2>&1
rc=$?
grep -aE "^(# |Total energy)" "$RAW"
echo "rc=$rc"; [ $rc -ne 0 ] && tail -15 "$RAW"
# A patch that does not install is silent; fail loudly rather than profile the
# wrong configuration (this cost 65 minutes once already).
for want in "d2h_view: view where" "unravel: assign_triu"; do
  grep -aqF "$want" "$RAW" || echo "PATCH DID NOT INSTALL: $want"
done
if [ -f "$W/$TAG.prof" ]; then
  cp "$W/$TAG.prof" "$R/" 2>/dev/null
  echo "--- top 30 by cumulative time ---"
  $A exec -B /data1 -B /tmp "$SIF" python -c "
import pstats, sys
p = pstats.Stats('$W/$TAG.prof')
p.sort_stats('cumulative').print_stats(30)" 2>&1 | sed -n '1,45p'
  echo "--- top 30 by time in the function itself ---"
  $A exec -B /data1 -B /tmp "$SIF" python -c "
import pstats
p = pstats.Stats('$W/$TAG.prof')
p.sort_stats('tottime').print_stats(30)" 2>&1 | sed -n '1,45p'
fi
rm -rf "$W/pb-cp"; echo

# ---------------- cell 2: CUDA-level, same size ----------------
NSYS=""
command -v nsys >/dev/null 2>&1 && NSYS=$(command -v nsys)
if [ -z "$NSYS" ]; then
  for m in nvhpc nvhpc-hpcx cuda; do
    module load "$m" >/dev/null 2>&1 && command -v nsys >/dev/null 2>&1 \
      && { NSYS=$(command -v nsys); break; }
  done
fi
[ -z "$NSYS" ] && for c in /shared/software/hpc_sdk/Linux_aarch64/*/compilers/bin/nsys; do
  [ -x "$c" ] && NSYS=$c && break
done
if [ -z "$NSYS" ]; then echo "NSYS_NOT_FOUND"; else
  echo "NSYS=$NSYS"
  SDK=$(echo "$NSYS" | sed 's#/Linux_aarch64/.*##')
  TAG=prof920_nsys; RAW=$L/raw_$TAG.log
  echo "=============== $TAG ==============="
  mkdir -p "$W/pb-ns"
  # nsys INSIDE the container with the SDK bound in: running it outside traces
  # only the host process, because CUDA tracing injects a preload library the
  # container's mount namespace cannot see.
  $A exec --nv -B /data1 -B "$SDK" -B /tmp \
     --env ${COMMON},PYBEST_TEMP=$W/pb-ns,TMPDIR=$W \
     "$SIF" "$NSYS" profile --trace=cuda,osrt,nvtx --cuda-memory-usage=true \
        --force-overwrite=true --output="$W/$TAG" \
        python $S/${SYS}.py > "$RAW" 2>&1
  rc=$?
  grep -aE "^(# |Total energy)" "$RAW"; echo "rc=$rc"
  [ $rc -ne 0 ] && tail -15 "$RAW"
  REP="$W/$TAG.nsys-rep"
  if [ -f "$REP" ]; then
    echo "--- trace size: $(du -h "$REP" | cut -f1) ---"
    for RPT in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum \
               cuda_api_sum osrt_sum; do
      echo "--- $RPT ---"
      $A exec -B /data1 -B "$SDK" -B /tmp "$SIF" \
        "$NSYS" stats --force-export=true --report "$RPT" --format table "$REP" 2>&1 \
        | grep -vE "^Processing|^Generating|^SQLite|^ *$" | head -20
    done
    cp "$REP" "$R/" 2>/dev/null && echo "kept $R/$TAG.nsys-rep"
  else
    echo "no .nsys-rep produced"
  fi
fi

rm -rf "$W"
echo "ALL DONE $(date)"
