#!/usr/bin/env python3
"""Generate benchmark geometries + PyBEST inputs for the RIKYU GPU study.

PyBEST ships 62 geometries but the largest is ~54 atoms (unit-test scale), so we
build our own ladder.

Two families:
  * (H2O)n water clusters -- reproduce the reference paper's EXACT dimensions
    (240 AO at cc-pVDZ, 580 AO at cc-pVTZ for n=10).
  * linear all-anti n-alkanes -- closed-shell with a large HOMO-LUMO gap, so RHF
    converges reliably. (Carbon clusters were rejected: multireference, so SCF
    convergence would become the experiment.)

Geometries are idealized, NOT optimized. That is correct for cost benchmarking,
where all that matters is that systems are physically reasonable and identical
across every configuration compared. The reference paper itself benchmarked
synthetic dimension-matched tensors for the same reason.

  python gen_systems.py --outdir systems
"""
from __future__ import annotations

import argparse
import math
import pathlib

# Idealized sp3 alkane internals (Angstrom / degrees).
D_CC, D_CH = 1.526, 1.090
ANG_CCC, ANG_HCH = 112.7, 107.0

# Water monomer, experimental geometry.
D_OH, ANG_HOH = 0.9572, 104.52
# O...O lattice spacing. Real H-bonds sit near 2.8 A; 3.2 A keeps a generated
# lattice clash-free and SCF-friendly. These clusters exist to match the
# reference paper's nbasis/nocc dimensions, not its energies.
D_OO = 3.2

# Basis functions per atom, cc-pVnZ (spherical harmonics).
# aug- adds one diffuse shell per angular momentum:
#   C/O  5s4p3d2f = 5 + 12 + 15 + 14 = 46      H  4s3p2d = 4 + 9 + 10 = 23
# Its max l stays 3 (f), the same as cc-pVTZ, so it asks nothing of libint that
# cc-pVTZ has not already exercised -- unlike cc-pVQZ, which needs g (l=4) and
# whose support depends on how the pre-generated libint tarball was built.
BF = {"cc-pvdz": {"C": 14, "H": 5, "O": 14},
      "cc-pvtz": {"C": 30, "H": 14, "O": 30},
      "aug-cc-pvtz": {"C": 46, "H": 23, "O": 46},
      "cc-pvqz": {"C": 55, "H": 30, "O": 55}}


def _norm(v):
    m = math.sqrt(sum(c * c for c in v))
    return [c / m for c in v]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def alkane(n: int):
    """All-anti C(n)H(2n+2): planar zig-zag backbone plus ideal hydrogens."""
    half = math.radians(ANG_CCC) / 2.0
    dx, dy = D_CC * math.sin(half), D_CC * math.cos(half)
    C = [[i * dx, (i % 2) * dy, 0.0] for i in range(n)]
    atoms = [("C", *c) for c in C]
    a_h = math.radians(ANG_HCH) / 2.0
    for i in range(n):
        nbrs = [j for j in (i - 1, i + 1) if 0 <= j < n]
        if len(nbrs) == 2:                                   # internal CH2
            u = _norm([C[nbrs[0]][k] - C[i][k] for k in range(3)])
            v = _norm([C[nbrs[1]][k] - C[i][k] for k in range(3)])
            bis = _norm([-(u[k] + v[k]) for k in range(3)])
            nrm = _norm(_cross(u, v))
            for sgn in (1, -1):
                d = _norm([bis[k] * math.cos(a_h) + sgn * nrm[k] * math.sin(a_h)
                           for k in range(3)])
                atoms.append(("H", *[C[i][k] + D_CH * d[k] for k in range(3)]))
        else:                                                # terminal CH3
            u = _norm([C[nbrs[0]][k] - C[i][k] for k in range(3)])
            e1 = _norm(_cross(u, [0.0, 0.0, 1.0]))
            e2 = _norm(_cross(u, e1))
            beta = math.radians(180.0 - 109.47)
            for t in (0.0, 2 * math.pi / 3, 4 * math.pi / 3):
                d = _norm([-u[k] * math.cos(beta)
                           + math.sin(beta) * (e1[k] * math.cos(t)
                                               + e2[k] * math.sin(t))
                           for k in range(3)])
                atoms.append(("H", *[C[i][k] + D_CH * d[k] for k in range(3)]))
    return atoms


def water_cluster(n: int):
    """(H2O)n on a clash-free cubic lattice, orientations alternated."""
    half = math.radians(ANG_HOH) / 2.0
    side = math.ceil(n ** (1.0 / 3.0))
    sites = [(i, j, k) for k in range(side) for j in range(side)
             for i in range(side)][:n]
    atoms = []
    for idx, (i, j, k) in enumerate(sites):
        ox, oy, oz = i * D_OO, j * D_OO, k * D_OO
        atoms.append(("O", ox, oy, oz))
        th = (idx % 4) * math.pi / 2.0            # avoid aligning all dipoles
        for sgn in (1, -1):
            lx, ly = D_OH * math.cos(half), sgn * D_OH * math.sin(half)
            atoms.append(("H", ox + lx * math.cos(th) - ly * math.sin(th),
                          oy + lx * math.sin(th) + ly * math.cos(th), oz))
    return atoms


def counts(atoms, basis: str):
    """(nbasis, nocc_total, ncore) with 1s frozen on C/N/O, as in the paper."""
    b = BF[basis]
    nb = sum(b[s] for s, *_ in atoms)
    elec = sum({"H": 1, "C": 6, "O": 8}[s] for s, *_ in atoms)
    heavy = sum(1 for s, *_ in atoms if s in ("C", "O"))
    return nb, elec // 2, heavy


def write_xyz(path: pathlib.Path, atoms, comment: str) -> None:
    lines = [str(len(atoms)), comment]
    lines += [f"{s} {x:14.8f} {y:14.8f} {z:14.8f}" for s, x, y, z in atoms]
    path.write_text("\n".join(lines) + "\n")


TEMPLATE = '''#!/usr/bin/env python3
"""Auto-generated PyBEST benchmark: {label}
nbasis={nb}  nocc={nocc}  ncore={ncore}  method={method}  basis={basis}
cd_threshold={thr}  maxiter={maxiter} (PYBEST_MAXITER overrides)

Metric follows Dobrowolska et al., JCTC 2026, 22, 6533: the mean time of ONE CC
iteration averaged over ~4 steps (vector function + amplitude update + energy).
We cap maxiter rather than converging -- convergence is not the measurement.
PYBEST_MAXITER overrides the cap: profiling runs use 2 so the trace stays
analysable, while the reported per-iteration metric is unaffected.

GPU is selected purely by environment; do not set it here:
  PYBEST_CUPY_AVAIL=1  (or PYBEST_PYTORCH_AVAIL=1)
  exactly one of PYBEST_C_SPLITTING=1 / PYBEST_X_SPLITTING=1
With none set, this is the CPU baseline.
"""
import os
import time

from pybest.gbasis import (
    compute_cholesky_eri, compute_kinetic, compute_nuclear,
    compute_nuclear_repulsion, compute_overlap, get_gobasis,
)
from pybest.linalg import CholeskyLinalgFactory
from pybest.occ_model import AufbauOccModel
from pybest.wrappers import RHF
from pybest.log import log
from pybest import filemanager

# PyBEST spills large cached arrays (e.g. exchange_oovv, ~10 GB at 920 AO) as
# HDF5 into filemanager.temp_dir, which defaults to "pybest-temp" RELATIVE TO
# THE CWD -- i.e. onto Lustre for a job run out of /data1. Rikyu gives each job
# 1.5 TB of node-local NVMe per requested GPU at /tmp, auto-deleted at job end,
# which is where these belong. The job script sets PYBEST_TEMP.
if os.environ.get("PYBEST_TEMP"):
    filemanager.temp_dir = os.environ["PYBEST_TEMP"]
print(f"# temp_dir={{filemanager.temp_dir}}", flush=True)

# Verbosity is raised to log.high only around the CC call; see the note there.

XYZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "{xyz}")
t0 = time.perf_counter()

basis = get_gobasis("{basis}", XYZ)
print(f"# nbasis={{basis.nbasis}} predicted={nb}", flush=True)

lf = CholeskyLinalgFactory(basis.nbasis)
occ_model = AufbauOccModel(basis, ncore={ncore})
orb_a = lf.create_orbital()

olp = compute_overlap(basis)
kin = compute_kinetic(basis)
ne = compute_nuclear(basis)
external = compute_nuclear_repulsion(basis)

# Setup caching. The Cholesky decomposition is the dominant setup cost and it
# does not depend on anything we vary between benchmark cells: 123 s at 580 AO
# and 628-1027 s at 920 AO, against a CCSD phase of 578 and 1827 s. Worse, it
# VARIES -- 565.7, 627.5, 628.6 and 1027.4 s were measured for identical work
# across four jobs -- which is a large part of why cross-job totals could not
# be compared. Caching it removes both the cost and the variance.
#
# The orbitals are cached too, but they are not used to bypass SCF: they are
# handed to RHF as its initial guess, so it converges in an iteration or two
# through the ordinary code path. Nothing is faked.
#
# Set PYBEST_SETUP_CACHE to a directory. The key covers geometry, basis and
# threshold, and the file is written atomically so a killed job cannot leave a
# half-written cache behind.
_CACHE_DIR = os.environ.get("PYBEST_SETUP_CACHE", "")
_cache_file = ""
if _CACHE_DIR:
    import hashlib

    # XYZ is already absolute (set above); the bare filename does not resolve,
    # because the job's working directory is not the systems/ directory.
    _key = hashlib.sha256(
        open(XYZ, "rb").read() + b"|{basis}|{thr}"
    ).hexdigest()[:16]
    _cache_file = os.path.join(_CACHE_DIR, f"setup_{{_key}}.h5")

_loaded = False
if _cache_file and os.path.exists(_cache_file):
    import h5py

    from pybest.linalg import CholeskyFourIndex

    t = time.perf_counter()
    try:
        with h5py.File(_cache_file, "r") as _f:
            if _f.attrs.get("nbasis") != basis.nbasis:
                raise ValueError(
                    f"cache nbasis {{_f.attrs.get('nbasis')}} != {{basis.nbasis}}")
            eri = CholeskyFourIndex.from_hdf5(_f["eri"])
            # assign_occupations() rejects a raw ndarray -- it requires a
            # DenseOneIndex -- so write through the array views instead, which
            # coeffs/energies/occupations expose directly.
            orb_a.coeffs[:] = _f["orb_coeffs"][:]
            orb_a.energies[:] = _f["orb_energies"][:]
            orb_a.occupations[:] = _f["orb_occupations"][:]
        _loaded = True
        print(f"# setup_cache=hit load_sec={{time.perf_counter()-t:.3f}} "
              f"nvec={{eri.nvec}} file={{_cache_file}}", flush=True)
    except Exception as _exc:
        print(f"# setup_cache=unusable ({{type(_exc).__name__}}: {{_exc}}), "
              f"recomputing", flush=True)
        _loaded = False

if not _loaded:
    t = time.perf_counter()
    eri = compute_cholesky_eri(basis, threshold={thr})
    print(f"# cholesky_eri_sec={{time.perf_counter()-t:.3f}}", flush=True)

t = time.perf_counter()
hf = RHF(lf, occ_model)
hf_out = hf(kin, ne, eri, external, olp, orb_a)
print(f"# rhf_sec={{time.perf_counter()-t:.3f}} "
      f"{{'(from cached guess)' if _loaded else ''}}", flush=True)

if _cache_file and not _loaded:
    import h5py

    t = time.perf_counter()
    _tmp = _cache_file + f".tmp{{os.getpid()}}"
    try:
        with h5py.File(_tmp, "w") as _f:
            _f.attrs["nbasis"] = basis.nbasis
            eri.to_hdf5(_f.create_group("eri"))
            _f["orb_coeffs"] = hf_out.orb_a.coeffs
            _f["orb_energies"] = hf_out.orb_a.energies
            _f["orb_occupations"] = hf_out.orb_a.occupations
        os.replace(_tmp, _cache_file)     # atomic: no half-written cache
        print(f"# setup_cache=written sec={{time.perf_counter()-t:.3f}} "
              f"size={{os.path.getsize(_cache_file)/2**30:.1f}} GiB", flush=True)
    except Exception as _exc:
        print(f"# setup_cache=write_failed ({{_exc}})", flush=True)
        if os.path.exists(_tmp):
            os.remove(_tmp)

# Silent CPU-fallback warnings are only emitted at log.high, so we need it --
# but PyBEST 2.2.0 CRASHES at log.do_high during SCF: the DIIS history logger
# does min(state.energy for state in ...) with energies still None and guards
# only EmptyData, so it escapes as TypeError (scf/scf_diis.py:480, reached from
# scf_diis.py:254). Raising verbosity only after SCF avoids that and still
# captures the CC contraction warnings, which are the ones that matter.
log.level = log.high

# Host-memory pinning experiment, applied in-process so the driver still runs
# directly (a runpy wrapper broke PyBEST's cache dump/load path whenever
# dump_cache was on, i.e. nact > 300). Nsight Systems (job 161307) found both
# backends spending more time on memory management than on arithmetic, in
# opposite ways: CuPy re-pins constantly because clean_memory flushes the pinned
# pool, while PyTorch never pins at all, so cudaMemcpyAsync to pageable memory
# is synchronous. PYBEST_PIN_STRATEGY selects the fix: pinned (staging buffer),
# cachedvram (freeze the free-VRAM probe), both, or baseline for neither.
_PIN_STRAT = os.environ.get("PYBEST_PIN_STRATEGY", "baseline")
print(f"# pin_strategy={{_PIN_STRAT}}", flush=True)
if _PIN_STRAT in ("pinned", "both"):
    import pybest.linalg._gpu_support as _gs

    if _gs.gpu_backend_select == "cupy":
        import cupy as _cp
        import numpy as _np

        _orig_get = _gs._ops["cupy"]["get_numpy"]
        _cpin: dict = {{}}

        def _cupy_via_pinned(x):
            try:
                n = int(x.size)
                e = _cpin.get(x.dtype)
                if e is None or e[0] < n:
                    sz = max(n, 2 * (e[0] if e else 0))
                    m = _cp.cuda.alloc_pinned_memory(sz * x.dtype.itemsize)
                    _cpin[x.dtype] = (sz, m,
                                      _np.frombuffer(m, dtype=x.dtype, count=sz))
                    e = _cpin[x.dtype]
                v = e[2][:n].reshape(x.shape)
                x.get(out=v)
                return v.copy()
            except Exception as _exc:
                print(f"# pin: cupy pinned path failed ({{_exc}})", flush=True)
                return _orig_get(x)

        _gs._ops["cupy"]["get_numpy"] = _cupy_via_pinned
        _gs._ops["cupy"]["as_numpy"] = _cupy_via_pinned
        print("# pin: cupy via a reused pinned buffer (bounded)", flush=True)
    elif _gs.gpu_backend_select == "pytorch":
        import torch as _t

        _pin: dict = {{}}

        def _via_pinned(x):
            n = x.numel()
            b = _pin.get(x.dtype)
            if b is None or b.numel() < n:
                b = _t.empty(max(n, 2 * (0 if b is None else b.numel())),
                             dtype=x.dtype, pin_memory=True)
                _pin[x.dtype] = b
            v = b[:n].view(x.shape)
            v.copy_(x, non_blocking=True)
            _t.cuda.synchronize()
            return v.numpy().copy()

        _gs._ops["pytorch"]["as_numpy"] = _via_pinned
        _gs._ops["pytorch"]["get_numpy"] = _via_pinned
        print("# pin: pytorch transfers via a reused pinned buffer", flush=True)

# crosslib_batching calls memory_usage() -- a cudaMemGetInfo round trip -- on
# every batched contraction, and the value it returns feeds a branch at
# crosslib_batching.py:702 (`> 0.4 * memhave`) that changes the batching
# strategy. Freezing the first reading removes the syscalls and keeps the batch
# plan stable instead of letting it drift as the allocator fills up.
if _PIN_STRAT in ("cachedvram", "both"):
    import pybest.linalg.crosslib_batching as _cb

    _orig_mu = _cb.memory_usage
    _mu: dict = {{}}

    def _cached_mu():
        if "v" not in _mu:
            _mu["v"] = _orig_mu()
            print(f"# cachedvram: frozen at {{_mu['v'] / 2**30:.2f}} GiB", flush=True)
        return _mu["v"]

    _cb.memory_usage = _cached_mu

# PyBEST flushes the PyTorch allocator from ~40 sites in crosslib_batching to
# make mem_get_info report the truth; at 920 AO that is 1059.8 s of empty_cache
# over 2399 calls. PYBEST_NOFLUSH corrects the estimate instead -- free plus the
# allocator's cached-but-free blocks -- so the flush has nothing left to do.
# Installed FIRST: it rebinds names inside crosslib_batching, which the other
# patches do not touch.
_NOFLUSH = os.environ.get("PYBEST_NOFLUSH", "")
if _NOFLUSH:
    import sys as _sys

    _sys.path.insert(0, os.environ.get(
        "PYBEST_BENCH_DIR", os.path.dirname(os.path.abspath(__file__))))
    import noflush_patch as _nf

    _nf.install(_NOFLUSH)

# The accumulate through PyTorch: torch.from_numpy(dest).add_(src, alpha=f) is
# a real fused axpy, threaded by ATen, zero-copy both ways. Only installs when
# PyTorch is the selected backend -- CuPy-only installs must not need torch.
if os.environ.get("PYBEST_ACCUM3") == "1":
    import sys as _sys

    _sys.path.insert(0, os.environ.get(
        "PYBEST_BENCH_DIR", os.path.dirname(os.path.abspath(__file__))))
    import accum3_patch as _a3

    _a3.install()

# The fused, threaded form of the accumulate. The first attempt (PYBEST_ACCUM)
# only moved the temporary and changed nothing, because the traffic is identical
# either way; the real problem is that numpy's += runs on one core at 13.7 GB/s
# against 768 GB/s of Grace memory bandwidth.
if os.environ.get("PYBEST_ACCUM2") == "1":
    import sys as _sys

    _sys.path.insert(0, os.environ.get(
        "PYBEST_BENCH_DIR", os.path.dirname(os.path.abspath(__file__))))
    import accum2_patch as _a2

    _a2.install()

# `arr[slice_] += factor * X` at twelve sites in base.py and once in
# td_GPU_helper allocates a full o^2 v^2 temporary -- 9.02 GiB at 920 AO -- and
# is 287 s, 16% of a 1827 s CCSD, in the own time of contract and td_GPU_helper.
# PYBEST_ACCUM makes the scaling happen in place on the array we already own.
if os.environ.get("PYBEST_ACCUM") == "1":
    import sys as _sys

    _sys.path.insert(0, os.environ.get(
        "PYBEST_BENCH_DIR", os.path.dirname(os.path.abspath(__file__))))
    import accum_patch as _ap

    _ap.install()

# Skip the staging copy for the 96% of transfers whose caller consumes the array
# immediately (job 162176: crosslib_batching.py:1487 alone carries 95.6-96.0% of
# D2H bytes, as `dest[...] += move_tensor_to_cpu(...)`). Installed AFTER the pin
# block so it replaces it; it does its own staging.
if os.environ.get("PYBEST_D2H_VIEW") == "1":
    import sys as _sys

    _sys.path.insert(0, os.environ.get(
        "PYBEST_BENCH_DIR", os.path.dirname(os.path.abspath(__file__))))
    import d2h_view_patch as _dv

    _dv.install()

# Attribute every device-to-host transfer to the PyBEST line that asked for it.
# Job 161947 measured 10.84 GB/s for a staged copy against 192.97 GB/s for a DMA
# into a registered destination; the sites that already know their destination
# could take the second path. This says how much volume those sites carry.
if os.environ.get("PYBEST_D2H_ATTRIB") == "1":
    import sys as _sys

    _sys.path.insert(0, os.environ.get(
        "PYBEST_BENCH_DIR", os.path.dirname(os.path.abspath(__file__))))
    import d2h_attribution as _da

    _da.install()

# RCCSD: unravel is 462 s at 920 AO and 144.6 s at 580 AO, reproducing to within
# 1% between backends because it is pure host work. Two of its three steps do
# far more than the arithmetic needs: assign_triu builds np.triu_indices per
# call (9.02 GiB of int64 indices at 920 AO) and iadd_transpose materialises two
# full temporaries for what is M += M.T on the (ov,ov) view. Job 161965 measured
# 13.7 s -> 2.5 s per unravel at 920 AO dimensions, bitwise identical.
if os.environ.get("PYBEST_UNRAVEL_FIX") == "1":
    import sys as _sys

    _sys.path.insert(0, os.environ.get(
        "PYBEST_BENCH_DIR", os.path.dirname(os.path.abspath(__file__))))
    import unravel_patch as _up

    _up.install()

_MAXITER = int(os.environ.get("PYBEST_MAXITER", "{maxiter}"))
if _MAXITER != {maxiter}:
    print(f"# maxiter overridden to {{_MAXITER}}", flush=True)

t = time.perf_counter()
{method_block}
dt = time.perf_counter() - t
# NB: this total includes the 4-index AO->MO transformation performed inside
# the CC call, so dt/maxiter is an UPPER BOUND, not the paper's metric. Take the
# per-iteration number from PyBEST's own "Iter ... Time" column above.
print(f"# {method}_total_sec={{dt:.3f}} upper_bound_per_iter={{dt/_MAXITER:.3f}} "
      f"maxiter={{_MAXITER}}", flush=True)
print(f"# total_sec={{time.perf_counter()-t0:.3f}}", flush=True)
# PyBEST prints its per-section timer table at exit (atexit -> log.print_footer);
# use it to separate CPU-only setup from the offloadable contractions.
'''

METHODS = {
    "pccd": ("from pybest.geminals import RpCCD\n"
             "cc = RpCCD(lf, occ_model)\n"
             "cc_out = cc(kin, ne, eri, hf_out, maxiter=_MAXITER)"),
    "ccsd": ("from pybest.cc import RCCSD\n"
             "cc = RCCSD(lf, occ_model)\n"
             "cc_out = cc(kin, ne, eri, hf_out, maxiter=_MAXITER)"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="systems")
    ap.add_argument("--threshold", type=float, default=1e-5,
                    help="cholesky threshold (paper 1e-5; PyBEST default 1e-4)")
    ap.add_argument("--maxiter", type=int, default=4,
                    help="CC iterations to time (paper averages 4-5)")
    args = ap.parse_args()
    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # REPRODUCE: land on the paper's dimensions so GB200 numbers are comparable.
    # EXTEND:    pass their 1004 ceiling -- GB200 has 184 GiB HBM vs GH200's ~96,
    #            and Rikyu grants up to 1600 GB host RAM vs their 480 GB.
    plan = [
        ("water", 10, "cc-pvdz", "ccsd", "reproduce"),   # their exact 240 AO
        ("water", 10, "cc-pvtz", "ccsd", "reproduce"),   # their exact 580 AO
        ("water", 20, "cc-pvdz", "ccsd", "reproduce"),
        ("alkane", 25, "cc-pvdz", "ccsd", "reproduce"),  # nocc~100 synthetic pt
        ("alkane", 37, "cc-pvdz", "ccsd", "reproduce"),  # nocc~150 synthetic pt
        # Third point on the C-split-share curve: same molecule and same nocc
        # as the 240 and 580 AO runs, so only nvirt grows (190 -> 530 -> 880).
        ("water", 10, "aug-cc-pvtz", "ccsd", "extend"),
        # Fourth point, 1150 AO. Still (H2O)10, so nocc stays at 40 active and
        # the series 240/580/920/1150 isolates basis size. Needs g functions
        # (l=4), which the pre-generated libint tarball may not have been built
        # with -- preflighted by jobs/run-ladder-large.sh before committing.
        ("water", 10, "cc-pvqz", "ccsd", "extend"),
        # Fallback if cc-pVQZ has no g functions: 1104 AO on aug-cc-pVTZ, which
        # is proven to work (920 AO ran fine). Costs a confound -- nocc active
        # moves 40 -> 48 -- so prefer cc-pVQZ when available.
        ("water", 12, "aug-cc-pvtz", "ccsd", "extend"),
        ("water", 20, "cc-pvtz", "ccsd", "extend"),
        ("alkane", 25, "cc-pvtz", "ccsd", "extend"),
        ("water", 30, "cc-pvtz", "ccsd", "extend"),
        ("alkane", 37, "cc-pvtz", "ccsd", "extend"),
        # pCCD companion: cheap per-iteration, so it sits deep in the batching
        # regime. One point only -- beyond ~2500 nbasis the CPU-only libchol
        # decomposition dominates the measurement.
        ("water", 20, "cc-pvqz", "pccd", "extend"),
    ]

    HBM_GIB = 184.0           # measured in-container
    PAPER_MAX_NBASIS = 1004   # their largest real system (L0 dye / cc-pVTZ)
    rows = []
    for family, n, basis, method, zone in plan:
        atoms, stem = ((water_cluster(n), f"h2o{n}") if family == "water"
                       else (alkane(n), f"c{n}h{2*n+2}"))
        nb, nocc, ncore = counts(atoms, basis)
        xyz = out / f"{stem}.xyz"
        if not xyz.exists():
            write_xyz(xyz, atoms, f"{family} n={n}")
        label = f"{stem}_{basis}_{method}"
        (out / f"{label}.py").write_text(TEMPLATE.format(
            label=label, nb=nb, nocc=nocc, ncore=ncore, method=method,
            basis=basis, thr=args.threshold, maxiter=args.maxiter,
            xyz=xyz.name, method_block=METHODS[method].format(maxiter=args.maxiter)))
        nvirt = nb - nocc
        gib = 5 * nb * nvirt * nvirt * 8 / 2**30   # (nchol,nvirt,nvirt), nchol~5N
        rows.append((label, zone, len(atoms), nb, nocc, ncore, nvirt, gib))

    print(f"{'system':<26}{'zone':<11}{'at':>4}{'nbasis':>8}{'nocc':>6}"
          f"{'ncore':>6}{'nvirt':>7}{'operand':>10}  note")
    for label, zone, na, nb, nocc, ncore, nvirt, gib in rows:
        note = ["BATCHES" if gib > HBM_GIB else "fits HBM"]
        if nb > PAPER_MAX_NBASIS:
            note.append("beyond paper's 1004")
        print(f"{label:<26}{zone:<11}{na:>4}{nb:>8}{nocc:>6}{ncore:>6}"
              f"{nvirt:>7}{gib:>9.1f}G  {', '.join(note)}")
    print(f"\n{len(rows)} inputs in {out}/  "
          f"(threshold={args.threshold}, maxiter={args.maxiter})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
