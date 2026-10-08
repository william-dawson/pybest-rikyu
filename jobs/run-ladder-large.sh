#!/bin/bash
# Why does PyTorch degrade at large N in the ladder contraction?
#
#   N      CuPy      vs GH200   PyTorch    vs GH200
#   800    235.18 s    1.30x     267.78 s    1.33x
#   900    397.97 s    1.24x     436.25 s    1.35x
#   1000   680.98 s    1.22x     684.50 s    1.18x
#   1100  1051.48 s    1.22x    1379.94 s    1.02x   <- PyTorch collapses
#
# Hypothesis, from crosslib_batching.py:503 and :1079 --
#   mem_gpu = memory_usage();  get_batch_sizes(chol_1, chol_2, mem_gpu)
#   while mem_need > mem_gpu * 0.9: add a batch
# memory_usage() is memGetInfo()[0] (CuPy) / torch.cuda.mem_get_info()[0]
# (PyTorch), i.e. DRIVER-level free VRAM, which excludes whatever each
# library's caching allocator holds. If PyTorch is sitting on more cached
# device memory at that moment, PyBEST hands it more and smaller batches on
# identical input -- more transfers, and worse as N grows.
#
# --log-batching prints the free-VRAM reading and the chosen batch counts, so
# the decision is directly comparable between backends.
#
# N=1100 runs a single pass purely for that comparison (we already have clean
# timings). N=1200 runs 3 passes, so it is also a new data point: their GH200
# column is n.c. there, and H100 C-split was 1864.7 s (PyTorch).
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
R=$P/results; mkdir -p "$R"
L=$P/logs;   mkdir -p "$L"

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
free -g | sed -n 2p
echo

# N:warmup:reps  -- 1100 is diagnostic only, 1200 is a real measurement
for SPEC in 1100:0:1 1200:1:1 1300:1:1; do
  N=${SPEC%%:*}; rest=${SPEC#*:}; W=${rest%%:*}; REPS=${rest##*:}
  for BACKEND in cupy pytorch; do
    if [ "$BACKEND" = cupy ]; then ENVV=PYBEST_CUPY_AVAIL=1
    else ENVV=PYBEST_PYTORCH_AVAIL=1; fi
    TAG="ladder_N${N}_${BACKEND}_Csplit"
    RAW=$L/raw_$TAG.log
    echo "=============== $TAG  (warmup=$W reps=$REPS) ==============="

    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw \
               --format=csv,noheader -l 30 > "$L/gpu_$TAG.csv" 2>/dev/null &
    SMI=$!

    BENCH_OUT=$R/$TAG.json \
    $A exec --nv -B /data1 \
       --env ${ENVV},PYBEST_C_SPLITTING=1,BENCH_OUT=$R/$TAG.json \
       "$SIF" python $P/repro_table4.py --nbasis "$N" --nocc 100 \
          --nvec-factor 5 --reps "$REPS" --warmup "$W" --log-batching \
          > "$RAW" 2>&1
    rc=$?

    kill $SMI 2>/dev/null; wait $SMI 2>/dev/null

    grep -aE "^(backend=|allocated|#batch|  (warmup|timed)|median|peak_rss|wrote|REFUSING|WARNING)" "$RAW"
    echo "rc=$rc"
    [ $rc -ne 0 ] && tail -20 "$RAW"
    # utilization.memory distinguishes "busy doing maths" from "busy moving data"
    awk -F, '{gsub(/ |%/,"",$2); gsub(/ |%/,"",$3);
              if ($2+0>g) g=$2+0; if ($3+0>m) m=$3+0}
             END{print "gpu_max_util="g"%  mem_ctrl_max_util="m"%"}' "$L/gpu_$TAG.csv"
    echo
  done
done
# Preflight for the next experiment: does this libint have g functions (l=4)?
# cc-pVQZ needs them, and (H2O)10/cc-pVQZ is 1150 AO -- the molecular point we
# want next, because it keeps nocc fixed and so extends 240/580/920 on the pure
# basis axis. Costs seconds, and tells us before we commit a multi-hour job.
echo "=============== preflight: cc-pVQZ (g functions) ==============="
$A exec --nv -B /data1 "$SIF" python -c "
from pybest.gbasis import get_gobasis, compute_cholesky_eri
from pybest.log import log
for b in ('cc-pvtz', 'aug-cc-pvtz', 'cc-pvqz'):
    try:
        g = get_gobasis(b, '$P/systems/h2o10.xyz', print_basis=False)
        print(f'  {b:<14} nbasis={g.nbasis}', flush=True)
    except Exception as e:
        print(f'  {b:<14} BASIS FAILED: {type(e).__name__}: {e}', flush=True)
        continue
    if b == 'cc-pvqz':
        try:
            eri = compute_cholesky_eri(g, threshold=1e-2)
            print(f'  {b:<14} cholesky OK, nchol={eri.array.shape[0]}', flush=True)
        except Exception as e:
            print(f'  {b:<14} CHOLESKY FAILED (likely libint max AM < 4): '
                  f'{type(e).__name__}: {e}', flush=True)
" 2>&1 | grep -aE "nbasis=|FAILED|cholesky OK" || echo "preflight produced no output"
echo

echo "ALL DONE $(date)"
