#!/bin/bash
# Does the PATCH work, as opposed to the monkey-patches it encodes?
#
# Everything measured so far used runtime monkey-patches. patches/
# pybest-2.2.0-perf.patch rewrites the same logic in the form it would take
# upstream, and has been checked for clean application, compilation and bitwise
# equivalence of the two numpy rewrites -- but no CCSD has ever executed through
# those exact files. Before sending it to the authors, run one.
#
# Method: bind-mount the three patched files over the container's installed
# PyBEST. Nothing else changes, and crucially NO monkey-patch env vars are set,
# so anything the patched cells gain comes from the patch itself.
#
# Expected, from the monkey-patch measurements:
#   240 AO   105.87 s -> ~62 s      (pinned + view + unravel)
#   580 AO   915.90 s -> ~569 s     -37.9%
# and energies -762.39447031 and -763.24436502 unchanged.
#
# The patch is SILENT -- unlike the monkey-patches it prints nothing -- so the
# job verifies the bind actually took effect by checking for a symbol that
# exists only in the patched source. A silent no-op has wasted an hour on this
# project once already.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
SIF=$P/pybest-rikyu-v3.sif
S=$P/systems
L=$P/logs; mkdir -p "$L"
PKG=/opt/venv/lib/python3.12/site-packages/pybest/linalg
BIND="-B $P/patched/_gpu_support.py:$PKG/_gpu_support.py"
BIND="$BIND -B $P/patched/crosslib_batching.py:$PKG/crosslib_batching.py"
BIND="$BIND -B $P/patched/dense_four_index.py:$PKG/dense/dense_four_index.py"
W=/tmp/pval-${SLURM_JOB_ID:-$$}

echo "host=$(hostname) nproc=$(nproc) job=${SLURM_JOB_ID:-none} date=$(date)"
echo

# ---- prove the bind works, before spending any GPU time on it ----
echo "=============== bind check ==============="
echo -n "stock:   "
$A exec --nv -B /data1 --env PYBEST_PYTORCH_AVAIL=1 "$SIF" python -c \
  "import pybest.linalg._gpu_support as g, pybest.linalg.crosslib_batching as c
print('BINDCHECK has_view_fn', hasattr(g,'move_tensor_to_cpu_view'),
      '| view_sites', open(c.__file__).read().count('move_tensor_to_cpu_view('))" 2>&1 | grep -a BINDCHECK
echo -n "patched: "
$A exec --nv -B /data1 $BIND --env PYBEST_PYTORCH_AVAIL=1 "$SIF" python -c \
  "import pybest.linalg._gpu_support as g, pybest.linalg.crosslib_batching as c
import pybest.linalg.dense.dense_four_index as d
print('BINDCHECK has_view_fn', hasattr(g,'move_tensor_to_cpu_view'),
      '| view_sites', open(c.__file__).read().count('move_tensor_to_cpu_view('),
      '| triu_rows', hasattr(d.DenseFourIndex,'_assign_triu_rows'),
      '| symm', hasattr(d.DenseFourIndex,'_iadd_own_transpose_square'))" 2>&1 | grep -a BINDCHECK
echo

for SYS in h2o10_cc-pvdz_ccsd h2o10_cc-pvtz_ccsd; do
  for MODE in stock patched; do
    if [ "$MODE" = patched ]; then USE="$BIND"; else USE=""; fi
    TAG="pval_${SYS}_${MODE}"; RAW=$L/raw_$TAG.log
    TMPD=$W-$SYS-$MODE; mkdir -p "$TMPD"
    echo "=============== $TAG ==============="
    /usr/bin/time -v $A exec --nv -B /data1 -B /tmp $USE \
       --env PYBEST_PYTORCH_AVAIL=1,PYBEST_C_SPLITTING=1,PYBEST_TEMP=$TMPD \
       "$SIF" python $S/${SYS}.py > "$RAW" 2>&1
    rc=$?
    grep -aE "^(# |Total energy|Correlation energy)" "$RAW"
    grep -aE "Maximum resident set size" "$RAW" || true
    sed -n '/Overview of CPU time usage/,/Page swaps/p' "$RAW" \
      | grep -aE "^(GPU: |Base: contract|RCCSD: unravel|Ints: CD-ERI|Total )" || true
    # No monkey-patch should be active in either cell: the patch replaces them.
    grep -aqE "^# (pin:|d2h_view:|unravel:|noflush:)" "$RAW" \
      && echo "UNEXPECTED: a monkey-patch is active, this cell is not a clean test"
    echo "rc=$rc"; [ $rc -ne 0 ] && tail -15 "$RAW"
    rm -rf "$TMPD"; echo
  done
done

echo "ALL DONE $(date)"
