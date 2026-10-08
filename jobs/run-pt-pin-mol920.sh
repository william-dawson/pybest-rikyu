#!/bin/bash
# The PyTorch pinning fix on a real CCSD at 920 AO -- the size where it should
# matter most, and the largest one with a published-style per-iteration metric.
#
# Why 920: the C-split share of PyTorch's GPU time rises with basis size
# (4.2% at 240 AO, 40.7% at 580, 65.6% at 920), and the pinning fix acts on
# exactly that share. 240 AO already gave -37.2% (job 161361) and 580 AO is
# running in job 161877, so this closes the series at the top end.
#
# Two cells, each ~80 min: the in-job baseline and `pinned`. An earlier version
# of this job ran `both` (pinned + cachedvram) as the candidate optimum; that is
# withdrawn. Freezing the free-VRAM reading does not remove overhead -- it
# changes the batch plan, sizing batches against the largest value the run will
# ever see -- so it is not a change to ship on a memory-constrained code, and
# measuring it at 80 min a cell buys nothing. See the pybest-gpu-offload skill.
#
# PYBEST_TEMP is node-local NVMe and NOT optional: PyBEST writes a checkpoint
# and reads it back with no fsync and no retry, so on Lustre the read fails
# nondeterministically (FileNotFoundError at cache.py:474). Every NVMe run has
# succeeded; Lustre runs are a coin flip. The driver also runs DIRECTLY -- a
# runpy wrapper broke the same cache path whenever dump_cache was on.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
S=$P/systems
L=$P/logs; mkdir -p "$L"
W=/tmp/ptmol-${SLURM_JOB_ID:-$$}

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
df -h /tmp | tail -1; free -g | head -2
echo

SYS=h2o10_aug-cc-pvtz_ccsd      # 920 AO, nocc 50, ncore 10, nvirt 870
for STRAT in baseline pinned; do
  TAG="ptpin_${SYS}_${STRAT}"; RAW=$L/raw_$TAG.log
  TMPD=$W-$STRAT; mkdir -p "$TMPD"
  echo "=============== $TAG ==============="
  nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used \
             --format=csv,noheader -l 30 > "$L/gpu_$TAG.csv" 2>/dev/null &
  SMI=$!
  $A exec --nv -B /data1 -B /tmp \
     --env PYBEST_PYTORCH_AVAIL=1,PYBEST_C_SPLITTING=1,PYBEST_TEMP=$TMPD,PYBEST_PIN_STRATEGY=$STRAT \
     "$SIF" python $S/${SYS}.py > "$RAW" 2>&1
  rc=$?
  kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
  grep -aE "^(# |Total energy|Correlation energy|T1 diagn)" "$RAW"
  # Their metric: the per-iteration time from PyBEST's own Iter/Time column.
  grep -aE "^ *[0-9]+ +-?[0-9]" "$RAW" | tail -6
  sed -n '/Overview of CPU time usage/,/Page swaps/p' "$RAW" \
    | grep -aE "^(GPU: |Base: contract|RCCSD: unravel|RCCSD: VecFct|Ints: CD-ERI|SCF |Total )" || true
  echo "rc=$rc"
  [ $rc -ne 0 ] && tail -15 "$RAW"
  awk -F, '{gsub(/ |%/,"",$2); gsub(/ |%/,"",$3);
            if ($2+0>g) g=$2+0; if ($3+0>m) m=$3+0}
           END{print "gpu_max_util="g"%  mem_ctrl_max_util="m"%"}' "$L/gpu_$TAG.csv"
  du -sh "$TMPD" 2>/dev/null; rm -rf "$TMPD"
  echo
done

echo "ALL DONE $(date)"
