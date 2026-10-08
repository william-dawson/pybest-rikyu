#!/bin/bash
# Add PyTorch (the other half of the CuPy-vs-PyTorch reproduction) and
# opt_einsum (PyBEST's contract() falls back to oe_contract in several paths;
# without it those fallbacks raise instead of degrading).
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/$USER/pybest
export APPTAINER_CACHEDIR=$P/.aptcache APPTAINER_TMPDIR=$P/.apttmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
SB=$P/.sbadd
log(){ echo "[$(date +%H:%M:%S)] $*"; }

rm -rf "$SB"
log "sif -> sandbox"
$A build --sandbox "$SB" "$P/pybest-rikyu-v2.sif" >"$P/logs/add-a.log" 2>&1 \
  || { echo "FATAL sandbox"; tail -10 "$P/logs/add-a.log"; exit 1; }

cat > "$SB/inner-add.sh" <<'INNER'
set -eux
export TMPDIR=/var/tmp
. /opt/venv/bin/activate
uv pip install opt_einsum
# aarch64 + CUDA torch. Try PyPI first (ships sbsa CUDA wheels), fall back to
# the cu128 index if the PyPI wheel turns out to be CPU-only.
uv pip install torch || true
python - <<'CHK'
import torch
print("TORCH_TRY", torch.__version__, "cuda", torch.cuda.is_available())
CHK
INNER

log "installing torch + opt_einsum"
$A exec --writable --fakeroot --no-mount tmp,home --env TMPDIR=/var/tmp \
   "$SB" bash /inner-add.sh >"$P/logs/add-b.log" 2>&1 \
  || { echo "FATAL install"; tail -25 "$P/logs/add-b.log"; exit 1; }
grep -a "TORCH_TRY" "$P/logs/add-b.log" || true

log "sandbox -> sif v3"
rm -f "$SB/inner-add.sh"
$A build "$P/pybest-rikyu-v3.sif" "$SB" >"$P/logs/add-c.log" 2>&1 \
  || { echo "FATAL sif"; tail -10 "$P/logs/add-c.log"; exit 1; }
rm -rf "$SB" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
log "DONE $(ls -lh $P/pybest-rikyu-v3.sif | awk '{print $5}')"
