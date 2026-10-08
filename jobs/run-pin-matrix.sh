#!/bin/bash
# Complete the bounded-pinning matrix: both backends x both workloads x sizes.
#
# What exists already (job 161368, PyTorch bounded; 161361, 240 AO):
#   ladder N=800   PyTorch 275.83 -> 228.45 s  (-17.2%)
#   ladder N=1100  PyTorch 1473.39 -> 1030.72 s (-30.0%)
#   CCSD 240 AO    PyTorch 97.62 -> 61.29 s    (-37.2%)
# and CuPy only in the UNBOUNDED form (ladder N=800, -3.4% for +43 GiB), which
# is the wrong shape of fix. This job measures CuPy BOUNDED and fills the
# molecular gaps that the Lustre checkpoint bug destroyed.
#
# Two rules learned the hard way:
#  * molecular cells set PYBEST_TEMP to node-local. PyBEST's checkpoint
#    dump/load fails NONDETERMINISTICALLY on Lustre (no fsync, no retry): at
#    580 AO, OMP=32 succeeded where 64, 16 and 8 failed identically.
#  * molecular cells run the driver DIRECTLY, using the PYBEST_PIN_STRATEGY
#    block now baked into the generated drivers. The ladder may use
#    pin_experiment.py since repro_table4.py has no CC object and no cache.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
R=$P/results; mkdir -p "$R"
L=$P/logs;   mkdir -p "$L"
W=/tmp/pinmx-${SLURM_JOB_ID:-$$}

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
nvidia-smi --query-gpu=index,name --format=csv,noheader; df -h /tmp | tail -1
echo

sample () { nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used \
              --format=csv,noheader -l 30 > "$1" 2>/dev/null & echo $!; }
maxutil () { awk -F, '{gsub(/ |%/,"",$2); gsub(/ |%/,"",$3);
                       if ($2+0>g) g=$2+0; if ($3+0>m) m=$3+0}
                      END{print "gpu_max_util="g"%  mem_ctrl_max_util="m"%"}' "$1"; }

# ---- ladder: CuPy bounded, the missing cells ----
for N in 800 1100; do
  for STRAT in baseline pinned; do
    TAG="mx_ladder_N${N}_cupy_${STRAT}"; RAW=$L/raw_$TAG.log
    echo "=============== $TAG ==============="
    SMI=$(sample "$L/gpu_$TAG.csv")
    BENCH_OUT=$R/$TAG.json \
    $A exec --nv -B /data1 \
       --env PYBEST_CUPY_AVAIL=1,PYBEST_C_SPLITTING=1,BENCH_OUT=$R/$TAG.json \
       "$SIF" python $P/pin_experiment.py \
          --driver $P/repro_table4.py --strategy $STRAT \
          -- --nbasis $N --nocc 100 --nvec-factor 5 --reps 1 --warmup 1 \
          > "$RAW" 2>&1
    rc=$?; kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
    grep -aE "^(# |backend=|  (warmup|timed)|median|peak_rss)" "$RAW"
    echo "rc=$rc"; [ $rc -ne 0 ] && tail -12 "$RAW"
    maxutil "$L/gpu_$TAG.csv"; echo
  done
done

# ---- molecular: both backends, both strategies, both sizes ----
for SYS in h2o10_cc-pvdz_ccsd h2o10_cc-pvtz_ccsd; do
  for BACKEND in cupy pytorch; do
    if [ "$BACKEND" = cupy ]; then AV=PYBEST_CUPY_AVAIL=1; else AV=PYBEST_PYTORCH_AVAIL=1; fi
    for STRAT in baseline pinned; do
      TAG="mx_${SYS}_${BACKEND}_${STRAT}"; RAW=$L/raw_$TAG.log
      TMPD=$W-$BACKEND-$STRAT; mkdir -p "$TMPD"
      echo "=============== $TAG ==============="
      SMI=$(sample "$L/gpu_$TAG.csv")
      $A exec --nv -B /data1 -B /tmp \
         --env ${AV},PYBEST_C_SPLITTING=1,PYBEST_TEMP=$TMPD,PYBEST_PIN_STRATEGY=$STRAT \
         "$SIF" python $P/systems/${SYS}.py > "$RAW" 2>&1
      rc=$?; kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
      grep -aE "^(# |Total energy|Correlation energy|T1 diagn)" "$RAW"
      sed -n '/Overview of CPU time usage/,/Page swaps/p' "$RAW" \
        | grep -aE "^(GPU: |Base: contract|RCCSD: unravel|RCCSD: VecFct|Ints: CD-ERI|SCF |Total )" || true
      echo "rc=$rc"; [ $rc -ne 0 ] && tail -12 "$RAW"
      maxutil "$L/gpu_$TAG.csv"; rm -rf "$TMPD"; echo
    done
  done
done
echo "ALL DONE $(date)"
