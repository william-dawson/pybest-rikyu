#!/bin/bash
# Re-profile at 920 AO with all four changes applied.
#
# The existing profile (job 162831) ran changes 1 to 3. Change 4 did not exist
# yet, and it targets the largest single item that profile found: the
# accumulate, at 34% of host-side work. So the host breakdown in README
# section 4 describes code that is no longer what ships.
#
# Two cells, both on the full stack:
#   cProfile  attributes host time to PyBEST functions, which is what the
#             section 4 breakdown table is built from
#   nsys      kernels, copies and CUDA API, for the kernel share and the
#             82%-of-DGEMM figure
#
# PYBEST_MAXITER=2 keeps the nsys trace analysable. Four iterations at ~740 s
# each produced roughly a gigabyte last time.
#
# Both cells assert that every requested patch announced itself. A stale
# driver on the cluster has silently disabled a patch three times, and it
# produces a plausible number rather than an error.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
S=$P/systems
L=$P/logs; mkdir -p "$L"
R=$P/results/profile920b; mkdir -p "$R"
W=/tmp/prof920b-${SLURM_JOB_ID:-$$}; mkdir -p "$W"
export TMPDIR=$W
SYS=h2o10_aug-cc-pvtz_ccsd
COMMON="PYBEST_PYTORCH_AVAIL=1,PYBEST_C_SPLITTING=1,PYBEST_D2H_VIEW=1,PYBEST_UNRAVEL_FIX=1,PYBEST_ACCUM3=1,PYBEST_BENCH_DIR=$P,PYBEST_MAXITER=2"

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
nvidia-smi --query-gpu=index,name --format=csv,noheader; df -h /tmp | tail -1
echo

# ---------------- cell 1: host-side attribution ----------------
TAG=prof920b_cprofile; RAW=$L/raw_$TAG.log
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
for want in "d2h_view: view where" "unravel: assign_triu" "accum3:"; do
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
  TAG=prof920b_nsys; RAW=$L/raw_$TAG.log
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
  # Same assertion as the cProfile cell: a patch that fails to install is
  # silent, and an unpatched trace would be indistinguishable from a patched
  # one until the numbers were already in the README.
  for want in "d2h_view: view where" "unravel: assign_triu" "accum3:"; do
    grep -aqF "$want" "$RAW" || echo "PATCH DID NOT INSTALL: $want"
  done
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
