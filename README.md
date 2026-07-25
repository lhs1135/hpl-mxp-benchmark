# HPL-AI Mixed-Precision Benchmark — Python (CPU / GPU / TTNN)

`hpl-ai.py` is a single-file Python port of the [HPL-MXP](https://hpl-mxp.org)
mixed-precision benchmark's reference implementation: generate a diagonally
dominant matrix, factor it with an unpivoted LU in single precision, then
recover full double-precision accuracy with GMRES iterative refinement. It
supports three interchangeable backends for the dominant O(N³)
factorization step — `cpu`, `gpu` (CUDA via PyTorch), and `ttnn`
(Tenstorrent Tensix) — selected with `--device` at run time, with
everything else (matrix generation, GMRES, the recursive panel
factorization) running identically on host across all three.

`hpl-ai.py` was ported from HPL-AI's C reference implementation, reproducing
its RNG (`matgen`/`vecgen`) bit-for-bit and its blocked LU algorithm
(`sgetrf_nopiv`, block size 32) structurally, so a given seed produces the
same matrix and the same factorization steps regardless of backend.

## Requirements

- Python 3.9+
- `numpy`, `scipy` — always required.
- `--device gpu` additionally requires `torch` built with CUDA support, and
  a CUDA-capable GPU.
- `--device ttnn` additionally requires `torch` and `ttnn` (Tenstorrent's
  TT-Metal Python stack), and a Tenstorrent device reachable at
  `device_id=0`.

`torch`/`ttnn` are imported lazily inside `main()`, after argument parsing —
`--help` and argument errors work with only `numpy`/`scipy` installed,
regardless of which `--device` was requested.

## Usage

```
python3 hpl-ai.py [N [MAX_IT]] [--device {cpu,gpu,ttnn}]
                   [--input-a A.bin [--input-b b.bin]]
                   [--metrics-file FILE]
```

- `N` — problem size (default 100).
- `MAX_IT` — GMRES restart length (default 50), capped at `N-1`.
- `--device {cpu,gpu,ttnn}` — backend for the LU factorization (default
  `cpu`). See [Where work runs](#where-work-runs) below.
- `--input-a FILE` — load `A` from a headerless raw binary file of `N*N`
  `float64` values in row-major (C) order, instead of generating it. `N` is
  inferred from the file size.
- `--input-b FILE` — load `b` from a raw binary file of `N` `float64`
  values (requires `--input-a`). If omitted, `b` is still generated, so you
  can pair a fixed `A` with a generated `b`.
- `--metrics-file FILE` — path to write the machine-readable metrics to
  (default `hpl-ai-metrics.txt` in the current directory, overwritten each
  run). See [Output](#output) below.

Run `python3 hpl-ai.py --help` for the full option reference with examples.

Matrix/vector generation (`matgen`/`vecgen`) uses the same LCG constants as
HPL-AI's C reference, so a given seed produces a bit-identical `A`/`b`.

## Where work runs

| Kernel | Placement | Precision | Rationale |
|---|---|---|---|
| `matgen` / `vecgen` | Host (Python) | fp64 | Seed is stateful — no inter-element parallelism |
| fp64 ↔ fp32 conversion | Host (NumPy) | — | Memory-bound; round-trip cost exceeds any device speedup |
| `_sgetrf2_nopiv` panel + internal strsm | Host (SciPy) | fp32 | Strictly serial recursive dependency, same on every backend |
| `L11⁻¹` computation (gpu/ttnn only) | Host (SciPy) | fp32 | 32×32 invert — negligible cost, enables block-row strsm as a device matmul |
| Block-row strsm `U12 = L11⁻¹·A12` | **cpu:** host (SciPy solve_triangular)  **gpu:** device (CUDA, torch.matmul)  **ttnn:** device (TTNN, matmul) | fp32 | On gpu/ttnn recast as a matmul against a precomputed `L11⁻¹` |
| Trailing sgemm `A22 -= A21·U12` | **cpu:** host (NumPy matmul)  **gpu:** device (CUDA)  **ttnn:** device (TTNN) | **cpu/gpu:** fp32  **ttnn:** bf16 | Dominant O(N³) cost; NB=32 matches the Tensix tile exactly on ttnn |
| Initial triangular solve | Host (SciPy) | fp32 | N×1 RHS gives a device no tile parallelism |
| GMRES (Arnoldi, preconditioner, Gram-Schmidt, Givens, convergence check) | Host (NumPy/SciPy) | fp64 | Requires fp64; neither Tensix nor the fp32 CUDA path used here has fp64 throughput worth offloading to |

`gpu` and `ttnn` share the same block structure (panel → invert pivot →
host↔device transfer → device strsm → device sgemm → transfer back), timed
identically. They differ in that `ttnn`'s sgemm operands go to bf16 for
Tensix throughput — which forces a bf16 round-trip of `U12` through the host
since it's also needed fp32 for the write-back — while `gpu`'s CUDA matmul
stays fp32 throughout with no such round-trip. See the docstrings on
`sgetrf_nopiv_cpu`, `sgetrf_nopiv_gpu`, and `sgetrf_nopiv_ttnn` in
`hpl-ai.py` for the exact per-block-column breakdown.

## Output

Stdout gets the human-readable report:

```
==============================================================================
              HPL-AI Mixed-Precision Benchmark  [Python]
==============================================================================
Backend: CPU (NumPy/SciPy)
Time: conversion to single             X.XXX ms
Time: LU factorization                 X.XXX ms
  ├─ panel  sgetrf2_nopiv  (host)      X.XXX ms  ( XX.X%)
  ├─ strsm  U12=L11⁻¹·A12 (host)       X.XXX ms  ( XX.X%)
  └─ sgemm  A22-=A21·U12  (host)       X.XXX ms  ( XX.X%)
Time: triangular solve                 X.XXX ms
Time: conversion to double             X.XXX ms
Residual norm at beginning of GMRES: X.XXXXXXe-XX
  GMRES iter   1: estimated residual = X.XXXXXXe-XX
  ...
Time: GMRES                            X.XXX ms
  GMRES converged in K / M iterations
Total time                             X.XXX ms
Effective GFLOPs                       X.XXXXXX

||Ax-b||_oo / ( eps * ( ||x||_oo * ||A||_oo + ||b||_oo ) * N )
eps = X.XXXXXXe-XX
Scaled residual = X.XXXXXX  ... PASSED

Machine-readable metrics written to: hpl-ai-metrics.txt
```

(`--device gpu`/`ttnn` print a six-row breakdown — panel, pivot-block
invert, host→device, device strsm, device sgemm, device→host — instead of
the three-row host-only breakdown above.)

The final scaled-residual check mirrors the C reference: `PASSED` if the
result is below the threshold of 16.0.

### Machine-readable metrics file

The same numbers are additionally written to `--metrics-file` (default
`hpl-ai-metrics.txt`) as one `key=value` pair per line — nothing else on
the line, no padding, no units suffix — so it's trivially parsed:

```python
metrics = dict(line.strip().split("=", 1) for line in open("hpl-ai-metrics.txt"))
```

It includes run parameters (`device`, `n`, `max_it`, `nb`), every timing
bucket from the human-readable report (`lu_panel_ms`, `lu_panel_pct`, ...),
the full GMRES residual trace (`gmres_iter_1_residual`, ...), and the final
`scaled_residual` / `scaled_residual_passed`. Floats are formatted with
`%.10g` (fixed or scientific, whichever is shorter) so every value
round-trips through `float()` regardless of magnitude; booleans are `1`/`0`.
The file is overwritten on every run — pass a different `--metrics-file`
per run if you want to keep a history.

## Known warnings

On some platforms you may see `RuntimeWarning: divide by zero / overflow /
invalid value encountered in matmul` during the LU factorization. This has
been traced to Apple's Accelerate BLAS/LAPACK backend (`numpy.show_config()`
reports `"blas": "accelerate"` on macOS): Accelerate's SGEMM kernel sets
spurious hardware FP exception flags internally even when every operand and
the result are small, finite, well-conditioned values — confirmed by
checking `np.isfinite` on the actual matmul operands/results at the point
of the warning, which show no NaN/Inf anywhere. It's cosmetic, not a
correctness issue; the scaled-residual check still passes. If it bothers
you, either filter it (`warnings.filterwarnings("ignore", message=".*encountered in matmul.*")`)
or install a NumPy build backed by OpenBLAS instead of Accelerate.
