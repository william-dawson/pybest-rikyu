#!/bin/bash
# Does cuBLAS FP64 emulation help PyBEST's real contraction, and at what cost?
#
# The microbenchmark (precision_probe.py) measured, on 8192^3 square DGEMM:
#     native  38.4 TFLOP/s   rel_err 1.56e-15
#     ADP     50.7           1.56e-15      1.32x
#     55 bit  70.6           4.87e-16      1.84x   <- faster AND more accurate
#     39 bit 111.4           1.61e-11      2.90x
#
# That is the best case. PyBEST issues batched, smaller, rectangular
# contractions, and cuBLAS falls back to native FP64 for shapes too small to
# benefit -- so the realised gain here should be lower. This measures by how
# much, on the exact contraction the reference paper optimised.
#
# N=800, nocc=100, nvec=5N, C-split. FP64 baselines from job 147057 (constant
# fill): CuPy 235.18 s, PyTorch 267.78 s. This job re-measures its own native
# baseline with random operands, because those are the only valid control.
#
# --random is mandatory throughout: ADP chooses its mantissa count from the
# data's dynamic range, so a constant fill would flatter it.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
R=$P/results; mkdir -p "$R"
L=$P/logs;   mkdir -p "$L"
# Node-local NVMe: the reference array is 36.5 GiB and gets re-read per mode.
SCRATCH=/tmp/emu-${SLURM_JOB_ID:-$$}; mkdir -p "$SCRATCH"

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
df -h /tmp | tail -1
echo

for BACKEND in cupy pytorch; do
  if [ "$BACKEND" = cupy ]; then AVAIL=PYBEST_CUPY_AVAIL=1
  else AVAIL=PYBEST_PYTORCH_AVAIL=1; fi
  REF=$SCRATCH/ref_$BACKEND.npy

  # mode label | cuBLAS env | extra args
  while IFS='|' read -r MODE CUENV EXTRA; do
    [ -z "$MODE" ] && continue
    TAG="emu_N800_${BACKEND}_${MODE}"
    RAW=$L/raw_$TAG.log
    echo "=============== $TAG  [$CUENV] ==============="

    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw \
               --format=csv,noheader -l 30 > "$L/gpu_$TAG.csv" 2>/dev/null &
    SMI=$!

    BENCH_OUT=$R/$TAG.json \
    $A exec --nv -B /data1 \
       --env ${AVAIL},PYBEST_C_SPLITTING=1,BENCH_OUT=$R/$TAG.json,${CUENV} \
       "$SIF" python $P/repro_table4.py --nbasis 800 --nocc 100 \
          --nvec-factor 5 --reps 2 --warmup 1 --random $EXTRA > "$RAW" 2>&1
    rc=$?

    kill $SMI 2>/dev/null; wait $SMI 2>/dev/null

    grep -aE "^(backend=|allocated|  (warmup|timed)|median|peak_rss|rel_err|wrote|REFUSING|WARNING)" "$RAW"
    echo "rc=$rc"
    [ $rc -ne 0 ] && tail -20 "$RAW"
    awk -F, '{gsub(/ |%/,"",$2); if ($2+0>m) m=$2+0} END{print "gpu_max_util="m"%"}' \
        "$L/gpu_$TAG.csv"
    echo
  done <<EOF
native|CUBLAS_EMULATE_DOUBLE_PRECISION=0|--ref-out $REF
adp|CUBLAS_EMULATE_DOUBLE_PRECISION=1|--ref-in $REF
m55|CUBLAS_EMULATE_DOUBLE_PRECISION=1,CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=55|--ref-in $REF
m39|CUBLAS_EMULATE_DOUBLE_PRECISION=1,CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT=39|--ref-in $REF
EOF

  rm -f "$REF"
done

echo "scratch_peak=$(du -sh "$SCRATCH" 2>/dev/null | cut -f1)"
rm -rf "$SCRATCH"
echo "ALL DONE $(date)"
