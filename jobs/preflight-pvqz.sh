#!/bin/bash
# Does this libint have g functions (l=4)? cc-pvqz.g94 ships with PyBEST, but
# max angular momentum is fixed when the pre-generated libint tarball is built,
# and PyBEST pulls that tarball from a third-party GitLab. Without l=4,
# (H2O)10/cc-pVQZ (1150 AO) is impossible and we fall back to
# (H2O)12/aug-cc-pVTZ (1104 AO).
#
# Seconds of work. Run standalone so the molecule job does not have to wait on
# the multi-hour ladder job to learn the answer.
set -uo pipefail
A=/shared/software/apptainer/bin/apptainer
P=/data1/rkp00012/rku00036/pybest
echo "host=$(hostname) date=$(date)"
$A exec --nv -B /data1 "$P/pybest-rikyu-v3.sif" python -c "
from pybest.gbasis import get_gobasis, compute_cholesky_eri
for b in ('cc-pvtz', 'aug-cc-pvtz', 'cc-pvqz'):
    try:
        g = get_gobasis(b, '$P/systems/h2o10.xyz')
    except Exception as e:
        print(f'RESULT {b:<14} BASIS FAILED {type(e).__name__}: {e}', flush=True)
        continue
    print(f'RESULT {b:<14} nbasis={g.nbasis}', flush=True)
    if b == 'cc-pvqz':
        # A loose threshold keeps this quick; we only need the integrals to be
        # evaluable at all, which is what a missing l=4 would prevent.
        try:
            eri = compute_cholesky_eri(g, threshold=1e-2)
            print(f'RESULT {b:<14} CHOLESKY OK nchol={eri.array.shape[0]} '
                  f'-- g functions present, 1150 AO is GO', flush=True)
        except Exception as e:
            print(f'RESULT {b:<14} CHOLESKY FAILED {type(e).__name__}: {e} '
                  f'-- fall back to h2o12_aug-cc-pvtz (1104 AO)', flush=True)
" 2>&1 | grep -aE "RESULT|Error|error" || echo "RESULT no output -- investigate"
echo "PREFLIGHT DONE $(date)"
