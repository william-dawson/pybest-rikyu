#!/bin/bash
# Patch the built container: bake library paths + add CUDA toolkit headers.
# The staged build made its sandbox from base.def (apt+uv only) and did the rest
# via `apptainer exec`, so the final .sif inherited base.def's EMPTY environment
# -- pybest-rikyu.def's %environment block was never applied. Hence
# "ImportError: libchol.so". Also cupy needs CTK headers at runtime for JIT.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/$USER/pybest
export APPTAINER_CACHEDIR=$P/.aptcache APPTAINER_TMPDIR=$P/.apttmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$P/logs"
SB=$P/.sbfix
log(){ echo "[$(date +%H:%M:%S)] $*"; }

rm -rf "$SB"
log "sif -> sandbox"
$A build --sandbox "$SB" "$P/pybest-rikyu.sif" >"$P/logs/fix-a.log" 2>&1 \
  || { echo "FATAL sandbox"; tail -15 "$P/logs/fix-a.log"; exit 1; }

cat > "$P/inner-fix.sh" <<'INNER'
set -eux
export TMPDIR=/var/tmp
printf '/opt/libint/lib\n/opt/libchol/lib\n' > /etc/ld.so.conf.d/pybest.conf
ldconfig
ldconfig -p | grep -E 'libchol|libint2' | head -5
. /opt/venv/bin/activate
uv pip install 'cupy-cuda13x[ctk]'
mkdir -p /.singularity.d/env
cat > /.singularity.d/env/91-pybest.sh <<'ENVEOF'
export LIBINT2_ROOT=/opt/libint
export LIBCHOL_ROOT=/opt/libchol
export LD_LIBRARY_PATH="/opt/libint/lib:/opt/libchol/lib:${LD_LIBRARY_PATH:-}"
export PATH="/opt/venv/bin:${PATH}"
export VIRTUAL_ENV=/opt/venv
ENVEOF
chmod +x /.singularity.d/env/91-pybest.sh
INNER

log "patching sandbox"
cp "$P/inner-fix.sh" "$SB/inner-fix.sh"
$A exec --writable --fakeroot --no-mount tmp,home --env TMPDIR=/var/tmp \
   "$SB" bash /inner-fix.sh \
   >"$P/logs/fix-b.log" 2>&1 \
  || { echo "FATAL patch"; tail -25 "$P/logs/fix-b.log"; exit 1; }
log "patch OK"

log "sandbox -> sif v2"
$A build "$P/pybest-rikyu-v2.sif" "$SB" >"$P/logs/fix-c.log" 2>&1 \
  || { echo "FATAL sif"; tail -15 "$P/logs/fix-c.log"; exit 1; }
rm -f "$SB/inner-fix.sh"
rm -rf "$SB" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$P/inner-fix.sh"
log "DONE $(ls -lh $P/pybest-rikyu-v2.sif | awk '{print $5}')"
