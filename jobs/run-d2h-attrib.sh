#!/bin/bash
# How much device-to-host VOLUME flows through call sites that could pass out=?
#
# Job 161947 priced the two paths: a staged copy, which is what PyBEST plus our
# pinned patch does, runs at 10.84 GB/s; the same DMA writing a registered
# destination runs at 192.97 GB/s. The gap is 18x and nsys cannot see it, because
# the copy is host-side work.
#
# crosslib_batching consumes a transfer three ways. At 437/489/998/1062/1487 it
# does `dest[...] += move_tensor_to_cpu(part)` and at 541/1117 it slice-assigns:
# both already know the destination, so the DMA could land there directly. At
# 351/582/925/1158 the array is RETURNED and escapes, which is why our patch has
# to copy. So the worth of recommending out= upstream is set by the share of
# volume the first group carries, and that is what this measures -- by caller
# line number, with no timing involved.
#
# Three sizes, both backends: the balance may shift with basis size, and CuPy
# routes through the same crosslib_batching code so the attribution applies to
# it too even though its transfers are already pinned.
#
# Cheap: 240 AO is ~2 min a cell, 580 AO ~25 min. PYBEST_TEMP node-local because
# PyBEST's checkpoint read-back fails nondeterministically on Lustre.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
S=$P/systems
L=$P/logs; mkdir -p "$L"
W=/tmp/d2hat-${SLURM_JOB_ID:-$$}

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
echo

for SYS in h2o10_cc-pvdz_ccsd h2o10_cc-pvtz_ccsd; do
  for BACKEND in pytorch cupy; do
    if [ "$BACKEND" = cupy ]; then AV=PYBEST_CUPY_AVAIL=1; else AV=PYBEST_PYTORCH_AVAIL=1; fi
    TAG="d2hat_${SYS}_${BACKEND}"; RAW=$L/raw_$TAG.log
    TMPD=$W-$SYS-$BACKEND; mkdir -p "$TMPD"
    echo "=============== $TAG ==============="
    $A exec --nv -B /data1 -B /tmp \
       --env ${AV},PYBEST_C_SPLITTING=1,PYBEST_TEMP=$TMPD,PYBEST_PIN_STRATEGY=pinned,PYBEST_D2H_ATTRIB=1,PYBEST_BENCH_DIR=$P \
       "$SIF" python $S/${SYS}.py > "$RAW" 2>&1
    rc=$?
    grep -aE "^# (d2h|nbasis|ccsd_total|total_sec|pin)" "$RAW"
    echo "rc=$rc"
    [ $rc -ne 0 ] && tail -12 "$RAW"
    rm -rf "$TMPD"; echo
  done
done

echo "ALL DONE $(date)"
