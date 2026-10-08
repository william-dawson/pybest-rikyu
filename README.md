# PyBEST on RIKYU (GB200)

Benchmarking the GPU offload path of [PyBEST](https://fizyka.umk.pl/~pybest/)
v2.2.0 on NVIDIA GB200, using RIKEN's AI4S machine RIKYU.

## Objective

Dobrowolska et al., *J. Chem. Theory Comput.* **2026**, *22*, 6533–6546
benchmark PyBEST's Cholesky-decomposed CCSD tensor contractions on H100 and
GH200. This repository measures the code on NVIDIA GB200, locates
the costly parts, and removes what can be removed.

Conditions throughout: PyTorch backend, C-split, Cholesky threshold 1e-5,
frozen core, one B200. "Unmodified" is PyBEST 2.2.0 as released; "patched" is
that code with the changes in section 2 applied. Each pair is measured in a
single job to try to reduce variance. 

## 1. Timings

### A complete CCSD

(H2O)10, seconds or minutes per CC iteration.

| AOs | basis | GH200 CuPy | GH200 PyTorch | GB200 CuPy | GB200 PyTorch | patched |
|---:|---|---:|---:|---:|---:|---:|
| 240 | cc-pVDZ | 23.9 s | 25.4 s | 21.03 s | 31.09 s | 12.20 s |
| 580 | cc-pVTZ | 5.5 m | 5.7 m | 4.39 m | 3.72 m | 2.07 m |
| 920 | aug-cc-pVTZ | -- | -- | 18.1 m | 15.97 m | 10.99 m |
| 1150 | cc-pVQZ | -- | -- | -- | -- | TBD |


The patched column is the PyTorch path with the four changes below applied,
measured against the GB200 PyTorch column in the same job.

Energies were checked against the unmodified runs to verify correctness. We
also measured memory usage to verify it does not increase.

### The ladder contraction

`xac,xbd,ecfd->efab` at the published dimensions: nocc = 100, nvec = 5N. This
is the contraction alone, with no SCF and no CC iteration, so of the four
changes only the pinned host buffer (change 1) applies.

| N | library | published | GB200 | ratio | with change 1 |
|---:|---|---:|---:|---:|---:|
| 800 | CuPy | 306.6 s (GH200) | 235.18 s | 1.30x | -- |
| 800 | PyTorch | 355.9 s (GH200) | 267.78 s | 1.33x | 233.06 s |
| 900 | CuPy | 495.0 s (GH200) | 397.97 s | 1.24x | -- |
| 900 | PyTorch | 587.1 s (GH200) | 436.25 s | 1.35x | -- |
| 1000 | CuPy | 829.1 s (GH200) | 680.98 s | 1.22x | -- |
| 1000 | PyTorch | 805.8 s (GH200) | 684.50 s | 1.18x | -- |
| 1100 | CuPy | 1282.7 s (GH200) | 1051.48 s | 1.22x | -- |
| 1100 | PyTorch | 1401.7 s (GH200) | 1379.94 s | 1.02x | 1030.72 s, 1.36x |
| 1200 | CuPy | 3042.4 s (H100) | 1622.32 s | 1.88x | -- |
| 1200 | PyTorch | 1864.7 s (H100) | 1948.04 s | 0.96x | 1583.43 s, 1.18x |
| 1300 | CuPy | 4665.3 s (H100) | 2325.18 s | 2.01x | -- |
| 1300 | PyTorch | 2679.1 s (H100) | 2909.81 s | 0.92x | 2236.56 s, 1.20x |

GH200 in the paper couldn't compute N = 1200 or 1300, so H100 is the reference
at those sizes.

At N = 1200 and 1300 unmodified PyTorch is slower on Blackwell than on Hopper.
The cause is unpinned host memory. CuPy already pins the memory so that
was why its performance did not degrade.

## 2. The changes

Changes 1, 2 and 4 affect the PyTorch path specifically, whereas  change 3 is 
backend-independent.

### 1. Pinned host buffer for device-to-host transfers

Before, in `linalg/_gpu_support.py`:

```python
"as_numpy": lambda t: t.cpu().numpy(),
```

`t.cpu()` allocates pageable host memory. `cudaMemcpyAsync` into a pageable
destination cannot use the DMA engine, so the driver stages the transfer
through a bounce buffer with a synchronous CPU memcpy. Device-to-host runs at
3.22 GB/s.

After this change, one pinned buffer per dtype, reused for the whole calculation:

```python
buffer = _PINNED_STAGING.get(tensor.dtype)
if buffer is None or buffer.numel() < count:
    buffer = torch.empty(count, dtype=tensor.dtype, pin_memory=True)
    _PINNED_STAGING[tensor.dtype] = buffer
view = buffer[:count].view(tensor.shape)
view.copy_(tensor, non_blocking=True)
```

Device-to-host now runs at 192.7 GB/s. This applies to the PyTorch path only.
CuPy already stages through a pinned pool and reaches 126 GB/s unaided.

### 2. Returning a view where a copy is not needed

Before, in `linalg/crosslib_batching.py`:

```python
result[tuple(view)] += move_tensor_to_cpu(outmat)
```

`move_tensor_to_cpu` copies out of the staging buffer, because the buffer is
reused and some callers keep the array they are given. This caller adds the
array into `result` and then drops it, so the copy is wasted. Recording which
line requested each transfer shows that 96% of all device-to-host bytes come
through this one.

After:

```python
result[tuple(view)] += move_tensor_to_cpu_view(outmat)
```

Seven call sites consume the array in place and take the view. Four return it
and keep the copying form.

### 3. A cheaper `unravel`

Before, in `linalg/dense/dense_four_index.py`:

```python
indtriu = np.triu_indices(self.nbasis * self.nbasis1, k)
self.array.reshape(n, m)[indtriu] = other.array[begin4:end4]
...
self.array[:] = self.array + self.array.transpose(transpose) * factor
```

`np.triu_indices` builds a pair of index arrays the size of the triangle
being filled, then scatters through them, when the rows of that triangle are
already contiguous slices. The last line builds two complete copies of the
array before storing the sum.

After, the triangle filled by rows, and the symmetrisation done in place:

```python
for i in range(n):
    length = n - i - k
    matrix[i, i + k:] = vector[offset : offset + length]
    offset += length
```

```python
# for an (o,v,o,v) array, (2,3,0,1) is matrix transpose on the (ov,ov) view,
# so this is M += M.T and blocks in place
for i0 in range(0, n, block):
    d = matrix[i0:i1, i0:i1]
    d += d.T.copy()
    for j0 in range(i1, n, block):
        upper = matrix[i0:i1, j0:j1].copy()
        matrix[i0:i1, j0:j1] = upper + matrix[j0:j1, i0:i1].T
        matrix[j0:j1, i0:i1] = matrix[i0:i1, j0:j1].T
```

`RCCSD: unravel` falls from 461.2 to 61.9 s at 920 AO.

### 4. A fused, threaded accumulate

Before, at twelve sites in `linalg/base.py` and once in `td_GPU_helper`:

```python
arr[slice_] += factor * td_helper(*args_)
```

`factor * X` builds a full temporary and the `+=` then reads it back, so each
accumulate makes three passes over the array where one would do. numpy's `+=`
also runs on a single core, which reaches 13.7 GB/s against the 384 GB/s a
Grace socket provides.

After, the transferred array records the scalar instead of applying it, so the
addition fuses into a single threaded pass:

```python
def __rmul__(self, other):          # factor * X records, does not scale
    self._factor = self._factor * other
    return self

# ... and the += that follows becomes
torch.from_numpy(dest).add_(torch.from_numpy(src), alpha=factor)
```

Each accumulate is now one pass, spread across sixteen threads.

## 3. Applying the changes

```bash
tar xf pybest.v2.2.0.tar.gz && cd pybest.v2.2.0
patch -p1 < ../patches/pybest-2.2.0-perf.patch
```

That applies changes 1–3, touching `linalg/_gpu_support.py`,
`linalg/crosslib_batching.py` and `linalg/dense/dense_four_index.py`.

Change 4 is a runtime patch, `bench/accum3_patch.py`, enabled with
`PYBEST_ACCUM3=1`. It is separate because it requires PyTorch for a numpy
operation, and PyBEST supports CuPy-only installations.

This patch is available for your convenience, but it may be better to simply
rewrite the changes in a smarter way based on the above guidance.

## 4. Where the time goes after patching

Everything below is per CC iteration, matching section 1, and measured on the
patched code.

| section, s per iteration | 240 AO | 580 AO | 920 AO |
|---|---:|---:|---:|
| `GPU: C-split` | 0.9 | 48.3 | 431.8 |
| `GPU: Generic` | 4.7 | 27.4 | 89.2 |
| `Base: contract` own | 4.1 | 25.4 | 76.2 |
| `RCCSD: unravel` | 1.0 | 5.7 | 15.5 |
| C-split share of GPU time | 16% | 64% | 83% |

`GPU: C-split` is the contraction the paper optimised. Changes 1, 2 and 4 act
on `GPU: Generic` and on the accumulate in `Base: contract`, and change 3 acts
on `RCCSD: unravel`. None of them touches C-split, which is why its share
reaches 83% once the others shrink, and why the overall reduction falls from
61% at 240 AO to 31% at 920.

Profiling at 920 AO splits the rest. GPU kernels run 501 s of each iteration
and reach 82% of the throughput a square DGEMM achieves on the same GPU, with
Nsight Compute putting them at about 98% of their own roofline, so no
arithmetic remains to be recovered. Of the remainder, the host spends 483 s
per iteration blocked waiting for the GPU and 344 s doing work of its own,
which divides as:

| | share of host work |
|---|---:|
| accumulate `arr[slice] += factor * X`, after change 4 | 21% |
| `clean_memory` | 20% |
| numpy `tensordot`, contractions that never reach the GPU | 12% |
| HDF5 checkpoint, written once per calculation | 11% |
| host-to-device overhead beyond the DMA | 8% |
| `unravel`, after change 3 | 6% |
| `reshape`, dense helpers, `copy` | 4% |
| unattributed | 18% |

For `clean_memory`, PyBEST calls it from about forty sites in
`crosslib_batching` regardless of how much GPU memory is free, so gating those
calls on actual scarcity is worth trying. Any attempt needs to watch GPU
memory as well as time, since those calls are what currently bounds it.

Peak GPU memory, unmodified against patched: 2.3 and 2.2 GiB at 240 AO, 85.8
and 85.6 at 580, 119.6 and 119.5 at 920.

Counting the whole process, including the Cholesky decomposition and SCF
before CCSD begins, CPU-side work is 67% at two iterations. Setup is paid
once, so that share falls with iteration count: 59% at four iterations, 52% at
ten, and 47% in the limit.

## Container

`container/pybest-rikyu.def` builds the benchmark image: PyBEST 2.2.0 with libint2
2.11.2 and libchol 0.1.12 (Cholesky ERI, 18 GPU contraction patterns), CuPy
14.2.0 and PyTorch 2.14.0+cu130, on aarch64. Build it with
`container/build-container.sh`; the resulting 4.8 GB image is not
stored here.

## License

Scripts in this repository are MIT. PyBEST, libint and libchol carry their
own licenses.
