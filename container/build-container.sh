#!/bin/bash
# Staged, resumable build of the PyBEST/RIKYU benchmarking container.
# Builds in node-local NVMe (/tmp) because libint creates thousands of small
# files and Lustre is poor at that; caches the expensive libint tree to /data1
# so a later-stage failure never forces a libint rebuild.
set -uo pipefail

PROJ=/data1/rkp00012/$USER/pybest
CACHE=$PROJ/cache; LOGS=$PROJ/logs
WORK=/tmp/pybest-build; SB=$WORK/sandbox
export APPTAINER_CACHEDIR=$WORK/aptcache APPTAINER_TMPDIR=$WORK/apttmp
mkdir -p "$CACHE" "$LOGS" "$WORK" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
die(){ echo "FATAL: $*"; exit 1; }
# TAR_OPTIONS: the libint (and libchol) tarballs record ownership as
# plgkbogusla/plgrid. Under --fakeroot only our uid <-> root is mapped, so GNU
# tar running as root defaults to --same-owner, fails the chown, and exits 2 —
# with the Makefile sending stderr to /dev/null, invisibly.
insb(){ apptainer exec --writable --fakeroot --no-mount tmp,home \
          --env TMPDIR=/var/tmp --env TAR_OPTIONS=--no-same-owner \
          "$SB" bash -c "export TMPDIR=/var/tmp TAR_OPTIONS=--no-same-owner; $1"; }

log "host=$(hostname) nproc=$(nproc) work=$WORK"
df -h /tmp | tail -1
ls -ld /tmp /var/tmp 2>&1

# ---------- Stage A: base sandbox (apt + uv) ----------
log "Stage A: base sandbox"
cat > "$WORK/base.def" <<'DEF'
Bootstrap: docker
From: ubuntu:24.04
%post
    set -eux
    export DEBIAN_FRONTEND=noninteractive
    # `apptainer build` has NO --no-mount, so the host /tmp is bind-mounted
    # into %post and apt-key cannot write there on a compute node. apt honours
    # TMPDIR / Dir::Temp, so route it to /var/tmp inside the image instead.
    mkdir -p /var/tmp && chmod 1777 /var/tmp
    export TMPDIR=/var/tmp
    apt-get -o Dir::Temp=/var/tmp update -qq
    apt-get -o Dir::Temp=/var/tmp install -y --no-install-recommends \
        ca-certificates wget curl git g++ gcc make cmake pkg-config \
        libboost-dev libboost-chrono-dev libgmp-dev \
        libeigen3-dev libopenblas-dev \
        python3 python3-dev autoconf automake libtool
    rm -rf /var/lib/apt/lists/*
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
    g++ --version | head -1
    ls -d /usr/include/eigen3 /usr/include/boost
DEF
apptainer build --fakeroot --sandbox "$SB" "$WORK/base.def" >"$LOGS/A-base.log" 2>&1 \
  || { tail -25 "$LOGS/A-base.log"; die "stage A"; }
log "Stage A OK"

# ---------- Stage B: PyBEST source ----------
log "Stage B: fetch PyBEST 2.2.0"
insb 'set -eux; mkdir -p /opt/src; cd /opt/src;
 wget -q https://www.fizyka.umk.pl/~pybest/downloads/pybest.v2.2.0.tar.gz;
 tar xzf pybest.v2.2.0.tar.gz; rm -f pybest.v2.2.0.tar.gz;
 ls -d /opt/src/pybest.v2.2.0' >"$LOGS/B-src.log" 2>&1 \
  || { tail -25 "$LOGS/B-src.log"; die "stage B"; }
log "Stage B OK"

# ---------- Stage C: libint2 (the long pole) ----------
if [ -f "$CACHE/libint-install.tar" ]; then
    log "Stage C: restoring cached libint tree"
    insb 'mkdir -p /opt/libint'
    tar -xf "$CACHE/libint-install.tar" -C "$SB/opt/libint" \
      || die "libint cache restore"
    log "Stage C restored from cache"
else
    log "Stage C: building libint2 2.11.2 (567 MB download + long compile)"
    # depends/Makefile pipes tar's stderr to /dev/null, which hid the real
    # error above. Let stderr through (keep stdout suppressed: it is -v noise).
    insb 'set -eux; cd /opt/src/pybest.v2.2.0/depends;
     sed -i "s#tar -xvf libint2.tar.gz > /dev/null 2>&1#tar -xvf libint2.tar.gz > /dev/null#" Makefile;
     sed -i "s#tar -xvf libchol.tar.gz > /dev/null 2>&1#tar -xvf libchol.tar.gz > /dev/null#" Makefile;
     grep -n "tar -xvf" Makefile' 
    # depends/Makefile has `CXX? = g++` (stray space => variable named "CXX?"),
    # so CXX is never set on Linux and -DCMAKE_CXX_COMPILER= goes out empty.
    insb 'set -eux; export CXX=g++ CXXFLAGS="-std=c++11 -O2";
     cd /opt/src/pybest.v2.2.0/depends;
     make libint -j'"$(nproc)"' LIBINT2_INSTALL_DIR=/opt/libint' \
     >"$LOGS/C-libint.log" 2>&1 \
      || { tail -40 "$LOGS/C-libint.log"; die "stage C (libint)"; }
    log "Stage C OK — caching libint tree to /data1"
    tar -cf "$CACHE/libint-install.tar" -C "$SB/opt/libint" . \
      && log "cached $(du -sh "$CACHE/libint-install.tar" | cut -f1)"
fi

log "libint layout:"
insb 'ls /opt/libint; echo "--- cmake dir:"; find /opt/libint -name "libint2-config.cmake" -o -name "Libint2Config.cmake" | head -5'

# libchol hardcodes -DLibint2_DIR=${LIBINT2_INSTALL_DIR}/lib/cmake/libint2.
# If libint landed in lib64 (or multiarch), bridge it with a symlink.
insb 'set -eux; if [ ! -d /opt/libint/lib/cmake/libint2 ]; then
   for d in /opt/libint/lib64 /opt/libint/lib/aarch64-linux-gnu; do
     if [ -d "$d/cmake/libint2" ]; then
       mkdir -p /opt/libint/lib
       ln -sfn "$d/cmake" /opt/libint/lib/cmake
       echo "bridged $d/cmake -> /opt/libint/lib/cmake"; break
     fi
   done
 else echo "lib/cmake/libint2 present, no bridge needed"; fi'

# ---------- Stage D: libchol (mandatory for the GPU path) ----------
log "Stage D: building libchol"
insb 'set -eux; export CXX=g++;
 export CXXFLAGS="-std=c++11 -O2 -I/usr/include/aarch64-linux-gnu/openblas-pthread";
 cd /opt/src/pybest.v2.2.0/depends;
 make libchol LIBINT2_INSTALL_DIR=/opt/libint LIBCHOL_INSTALL_DIR=/opt/libchol \
   OpenBLAS_INCLUDE_FILE=cblas.h' >"$LOGS/D-libchol.log" 2>&1 \
  || { tail -40 "$LOGS/D-libchol.log"; die "stage D (libchol)"; }
insb 'ls /opt/libchol/lib /opt/libchol/include 2>&1 | head'
log "Stage D OK"

# ---------- Stage E: Python layer + PyBEST wheel ----------
log "Stage E: python layer, pybest wheel, cupy"
insb 'set -eux; cd /opt/src/pybest.v2.2.0;
 sed -i "/^PySide6/d" requirements.txt;
 export UV_CACHE_DIR=/var/cache/uv XDG_CACHE_HOME=/var/cache;
 uv venv /opt/venv --python 3.12;
 . /opt/venv/bin/activate;
 uv pip install -r requirements-build.txt;
 uv pip install -r requirements.txt;
 uv pip install "build~=1.0.3";
 export LIBINT2_ROOT=/opt/libint LIBCHOL_ROOT=/opt/libchol;
 export LD_LIBRARY_PATH=/opt/libint/lib:/opt/libchol/lib:${LD_LIBRARY_PATH:-};
 python -m build --wheel --no-isolation;
 uv pip install pybest --find-links dist/ --no-index;
 uv pip install cupy-cuda13x' >"$LOGS/E-python.log" 2>&1 \
  || { tail -40 "$LOGS/E-python.log"; die "stage E (python)"; }
log "Stage E OK"

# ---------- Stage F: verify inside sandbox ----------
log "Stage F: verification"
insb 'set -eux; . /opt/venv/bin/activate;
 export LD_LIBRARY_PATH=/opt/libint/lib:/opt/libchol/lib:${LD_LIBRARY_PATH:-};
 python -c "import pybest; print(\"pybest\", pybest.__version__)";
 python -c "import pybest.core as c; print(\"cholesky compiled in:\", any(\"cholesky\" in d for d in dir(c)))";
 python -c "from pybest.linalg._gpu_support import gpu_contraction_optimized as g; print(\"gpu patterns:\", len(g))";
 python -c "import cupy; print(\"cupy\", cupy.__version__)"' 2>&1 | tail -20
log "Stage F done"

# ---------- Stage F2: slim the sandbox before freezing ----------
# /opt/src holds the unpacked 567 MB libint source + its build tree (many GB).
# The install prefixes (/opt/libint, /opt/libchol) and /opt/venv are what we
# actually need, so keep the built wheel and drop the rest.
log "Stage F2: slimming sandbox"
insb 'set -eux; mkdir -p /opt/wheels;
 cp /opt/src/pybest.v2.2.0/dist/*.whl /opt/wheels/ 2>/dev/null || true;
 rm -rf /opt/src /var/cache/uv /var/cache/apt /root/.cache;
 du -sh /opt/libint /opt/libchol /opt/venv /opt/wheels 2>&1'
log "Stage F2 OK"

# ---------- Stage G: freeze to .sif on /data1 ----------
log "Stage G: converting sandbox -> .sif"
apptainer build --fakeroot "$WORK/pybest-rikyu.sif" "$SB" >"$LOGS/G-sif.log" 2>&1 \
  || { tail -25 "$LOGS/G-sif.log"; die "stage G (sif)"; }
cp "$WORK/pybest-rikyu.sif" "$PROJ/pybest-rikyu.sif" || die "copy sif to /data1"
log "DONE: $PROJ/pybest-rikyu.sif ($(du -sh "$PROJ/pybest-rikyu.sif" | cut -f1))"
