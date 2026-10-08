#!/bin/bash
# The headline table, measured properly: stock against the full patch stack,
# each pair inside one job, at three basis sizes.
#
# Everything so far has been built from in-job ratios chained across separate
# jobs, because absolute totals drift (Ints: CD-ERI alone varied 565.7 to
# 1027.4 s for identical work). That was defensible for individual patches but
# it is not how the final number should be quoted. Here each size is one job
# cell pair, so the ratio needs no chaining.
#
# Full stack, PyTorch only:
#   1 pinned host buffer          PYBEST_D2H_VIEW implies it
#   2 cheap unravel               PYBEST_UNRAVEL_FIX
#   3 view instead of copy        PYBEST_D2H_VIEW
#   4 fused threaded accumulate   PYBEST_ACCUM3 (16 threads by default; 16 beat
#                                 64 at 580 AO, 494.80 against 508.43 s)
#
# Patch 5 now also covers base.py's twelve accumulate sites: the first version
# demanded C-contiguous destinations and so fused only td_GPU_helper, leaving
# Base: contract's own time untouched at 105.9 s. torch.from_numpy accepts any
# positive-strided array, so the guard now asks only for that.
#
# The energy is the acceptance test, not the timing. VERIFY=20 additionally
# checks the first twenty fused accumulates against numpy at runtime.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
S=$P/systems
L=$P/logs; mkdir -p "$L"
W=/tmp/final-${SLURM_JOB_ID:-$$}
# NOFLUSH is deliberately NOT here. Skipping the allocator flush is worth
# -5.9% at 580 AO and -7.4% at 920, but sampling GPU0 during job 163891 showed
# what it costs: peak VRAM 119.5 -> 146.8 GiB and MEAN 30.1 -> 94.8 GiB, a
# factor of 3.1, on a 184 GiB card. Scaled to 1150 AO, where o^2 v^2 goes
# 9.02 -> 15.5 GiB, that projects past the card. The idea is not dead -- the
# flush is indiscriminate rather than useless, and flushing only when memory is
# actually scarce would keep most of the gain -- but it is not shipping today.
FULL="PYBEST_D2H_VIEW=1,PYBEST_UNRAVEL_FIX=1,PYBEST_ACCUM3=1,PYBEST_ACCUM3_VERIFY=20"

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
nvidia-smi --query-gpu=index,name --format=csv,noheader; echo

# system : reference energy at maxiter=4
for PAIR in "h2o10_cc-pvdz_ccsd:-762.39447031" \
            "h2o10_cc-pvtz_ccsd:-763.24436502" \
            "h2o10_aug-cc-pvtz_ccsd:-763.32762373"; do
  SYS=${PAIR%%:*}; REF=${PAIR##*:}
  for MODE in stock full; do
    if [ "$MODE" = full ]; then EXTRA=",$FULL"; else EXTRA=""; fi
    TAG="final_${SYS}_${MODE}"; RAW=$L/raw_$TAG.log
    TMPD=$W-$SYS-$MODE; mkdir -p "$TMPD"
    echo "=============== $TAG ==============="
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 20 \
               > "$L/gpu_$TAG.csv" 2>/dev/null &
    SMI=$!
    /usr/bin/time -v $A exec --nv -B /data1 -B /tmp \
       --env PYBEST_PYTORCH_AVAIL=1,PYBEST_C_SPLITTING=1,PYBEST_TEMP=$TMPD,PYBEST_BENCH_DIR=$P${EXTRA} \
       "$SIF" python $S/${SYS}.py > "$RAW" 2>&1
    rc=$?
    kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
    grep -aE "^(# |Total energy)" "$RAW"
    grep -aE "Maximum resident set size" "$RAW" || true
    sed -n '/Overview of CPU time usage/,/Page swaps/p' "$RAW" \
      | grep -aE "^(GPU: |Base: contract|RCCSD: unravel|Ints: CD-ERI|Total )" || true
    # -F and -- are both required: the reference starts with a minus, which
    # grep would otherwise parse as an option bundle. That bug made an earlier
    # job report ENERGY WRONG on every cell, including correct ones.
    if grep -aqF -- "$REF" "$RAW"; then
      echo "ENERGY OK ($REF)"
    else
      echo "!!! ENERGY WRONG -- this cell's timing is meaningless"
      grep -a "Total energy" "$RAW" | tail -1
    fi
    if [ "$MODE" = full ]; then
      for want in "d2h_view: view where" "unravel: assign_triu" "accum3:"; do
        grep -aqF "$want" "$RAW" || echo "PATCH DID NOT INSTALL: $want"
      done
      grep -aE "^# accum3: fused" "$RAW" || true
    fi
    echo "rc=$rc"; [ $rc -ne 0 ] && tail -12 "$RAW"
    awk -F, "{gsub(/ MiB/,\"\",\$2); gsub(/ /,\"\",\$1);
              if(\$1==0){n++; s+=\$2; if(\$2+0>m)m=\$2+0}}
             END{if(n) printf \"vram_peak=%.1f GiB  vram_mean=%.1f GiB\n\", m/1024, s/n/1024}" \
        "$L/gpu_$TAG.csv"
    rm -rf "$TMPD"; echo
  done
done

echo "ALL DONE $(date)"
