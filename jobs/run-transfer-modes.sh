#!/bin/bash
# Price every host<->device transfer mechanism GB200 offers, before wiring any
# of them into PyBEST. Cheap: one GPU, ~10 minutes, no PyBEST, no CCSD.
#
# The anomaly that motivates it, from the Nsight Systems trace of a 240 AO CCSD
# (job 161307): H2D ran at 128 GB/s out of ordinary pageable numpy memory while
# D2H ran at 3.1 GB/s into ordinary pageable numpy memory. A 41x asymmetry with
# the same kind of host buffer at both ends is not explained by "pageable copies
# are staged" alone, and on a Grace platform the GPU can also reach host memory
# coherently over NVLink-C2C. Which mechanism is in play decides whether the
# right fix is registering PyBEST's arrays (cheap patch) or allocating its
# outputs in managed memory (a change for the authors).
#
# Two sizes: 2 GiB is the scale of a single C-split batch, 16 GiB is past any
# cache or staging-buffer effect.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
R=$P/results/transfer; mkdir -p "$R"
L=$P/logs; mkdir -p "$L"

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# C2C is the thing being measured, so record what the driver thinks it has.
nvidia-smi --query-gpu=driver_version --format=csv,noheader
echo

for G in 2 16; do
  TAG="transfer_modes_${G}gib"
  echo "=============== $TAG ==============="
  BENCH_OUT=$R/$TAG.json TM_GIB=$G TM_REPS=5 \
  $A exec --nv -B /data1 \
     --env TM_GIB=$G,TM_REPS=5,BENCH_OUT=$R/$TAG.json \
     "$SIF" python $P/transfer_modes.py 2>&1 | tee "$L/raw_$TAG.log"
  echo "rc=${PIPESTATUS[0]}"
  echo
done

echo "ALL DONE $(date)"
