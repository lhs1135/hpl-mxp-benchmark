#!/usr/bin/env python3
"""
HPL-AI Mixed-Precision Benchmark — Python implementation, CPU/GPU/TTNN backends.

Host  (Python + NumPy + SciPy, fp32/fp64):
  matgen / vecgen, panel factorization (_sgetrf2_nopiv),
  initial triangular solve, GMRES — identical on all three backends.

--device cpu (default):
  Blocked LU (sgetrf_nopiv_cpu) runs entirely on host, fp32 — a direct port
  of the C reference's sgetrf_nopiv/strsm/sgemm (blas.c, sgetrf_nopiv.c).
  No torch/ttnn dependency, no accelerator required.

--device gpu:
  Blocked LU (sgetrf_nopiv_gpu) offloads to a CUDA GPU via PyTorch:
  Block-row strsm and trailing sgemm both run as torch.matmul on-device,
  fp32 throughout. Requires torch built with CUDA support.

--device ttnn:
  Blocked LU (sgetrf_nopiv_ttnn) offloads to a Tenstorrent Tensix via TTNN:
  Block-row strsm in sgetrf_nopiv_ttnn (U12 = L11_inv @ A12)     — fp32.
  Trailing sgemm in sgetrf_nopiv_ttnn  (A22 -= A21 @ U12)        — bf16,
  fed by an on-device ttnn.typecast of U12 (fp32 -> bf16, no PCIe hop —
  see test_ttnn_typecast.py).

Run with --help for CLI usage.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.linalg

# torch/ttnn are imported lazily in main(), after argument parsing, so that
# --help and argument errors exit without touching the TT device stack
# (importing ttnn can itself trigger driver/firmware init as a side effect).

# ─── LCG random number generator ─────────────────────────────────────────────
# Constants must match the C reference exactly so the generated matrix
# is identical (uint64 overflow = implicit mod 2^64).
_LCG_A   = np.uint64(6364136223846793005)
_LCG_C   = np.uint64(1)
_LCG_MUL = 5.4210108624275222e-20   # 2^-64: maps uint64 → [0, 1)

def _lcg_step(seed: np.ndarray) -> np.uint64:
    seed[0] = seed[0] * _LCG_A + _LCG_C
    return seed[0]

def _lcg_double(seed: np.ndarray) -> float:
    return float(_lcg_step(seed)) * _LCG_MUL - 0.5


def matgen(n: int, iseed: int) -> np.ndarray:
    """Row-diagonally-dominant fp64 matrix, identical to C matgen."""
    A    = np.zeros((n, n), dtype=np.float64)
    diag = np.zeros(n,      dtype=np.float64)
    seed = np.array([np.uint64(iseed)], dtype=np.uint64)
    # Advance seed column-major (j outer, i inner) to match C loop order.
    for j in range(n):
        for i in range(n):
            v        = _lcg_double(seed)
            A[i, j]  = v
            diag[i] += abs(v)
    for i in range(n):
        A[i, i] = diag[i] - abs(A[i, i])
    return A


def vecgen(n: int, iseed: int) -> np.ndarray:
    v    = np.zeros(n, dtype=np.float64)
    seed = np.array([np.uint64(iseed)], dtype=np.uint64)
    for i in range(n):
        v[i] = _lcg_double(seed)
    return v


# ─── Panel factorization: host, fp32, recursive unblocked ────────────────────

def _sgetrf2_nopiv(A: np.ndarray) -> None:
    """
    Recursive LU without pivoting (in-place on a fp32 numpy view).

    Mirrors C sgetrf2_nopiv exactly:
      base case n==1  : column-scale below pivot
      recursive case  : split at n//2, factor left half, strsm right half,
                        sgemm trailing update, recurse bottom-right.

    The strsm here is always ≤ 16×16 (halves at every level), so SciPy on
    the host is appropriate — no device offload worthwhile.
    """
    m, n = A.shape
    if m <= 1 or n == 0:
        return
    if n == 1:
        A[1:, 0] /= A[0, 0]
        return
    n1 = min(m, n) // 2
    n2 = n - n1

    _sgetrf2_nopiv(A[:, :n1])

    # strsm: L[:n1, :n1] * U12 = A[:n1, n1:]  (≤ 16×16)
    # scipy ignores the upper triangle and diagonal (unit_diagonal=True).
    A[:n1, n1:] = scipy.linalg.solve_triangular(
        A[:n1, :n1], A[:n1, n1:], lower=True, unit_diagonal=True,
    )

    # sgemm: A[n1:, n1:] -= L21 @ U12
    A[n1:, n1:] -= A[n1:, :n1] @ A[:n1, n1:]

    _sgetrf2_nopiv(A[n1:, n1:])


# ─── Blocked LU ───────────────────────────────────────────────────────────────

NB = 32   # block size == Tensix tile dimension — exact match, no padding needed


def sgetrf_nopiv_cpu(A: np.ndarray) -> dict[str, float]:
    """
    Blocked LU without pivoting, in-place on A (n×n numpy fp32 array).

    Direct port of the C reference's sgetrf_nopiv (sgetrf_nopiv.c) with its
    strsm/sgemm (blas.c) inlined as NumPy/SciPy calls — host-only, fp32,
    nb=32. No device offload; this is the CPU baseline.

    Per block column j:
      Panel factorization (_sgetrf2_nopiv) : fp32 — serial recursive, (N-j)×32
      Block-row strsm  U12 = L11^-1 * A12  : fp32 — triangular solve, matches C strsm
      Trailing sgemm   A22 -= A21 @ U12    : fp32 — dominant O(N^3) cost

    Returns a dict of accumulated wall-clock seconds per phase.
    """
    tm = dict(panel=0.0, strsm=0.0, sgemm=0.0)
    _t = time.perf_counter

    n = A.shape[0]
    for j in range(0, n, NB):
        jb = min(NB, n - j)

        # ── panel factorization ───────────────────────────────────────────
        t0 = _t(); _sgetrf2_nopiv(A[j:, j:j+jb]); tm["panel"] += _t() - t0

        if j + jb >= n:
            break

        # ── block-row strsm: L11 * U12 = A12 ──────────────────────────────
        t0 = _t()
        A[j:j+jb, j+jb:] = scipy.linalg.solve_triangular(
            A[j:j+jb, j:j+jb], A[j:j+jb, j+jb:], lower=True, unit_diagonal=True,
        )
        tm["strsm"] += _t() - t0

        # ── trailing sgemm: A22 -= A21 @ U12 ──────────────────────────────
        t0 = _t()
        A[j+jb:, j+jb:] -= A[j+jb:, j:j+jb] @ A[j:j+jb, j+jb:]
        tm["sgemm"] += _t() - t0

    return tm

def _to_cuda(arr: np.ndarray, device) -> "torch.Tensor":
    """numpy → contiguous fp32 torch tensor on a CUDA device."""
    t = torch.from_numpy(np.ascontiguousarray(arr))
    return t.to(device=device, dtype=torch.float32)


def _from_cuda(t: "torch.Tensor") -> np.ndarray:
    """CUDA torch tensor → numpy (fp32). .cpu() is the sync point (d2h)."""
    return t.cpu().numpy()


def sgetrf_nopiv_gpu(A: np.ndarray, device) -> dict[str, float]:
    """
    Blocked LU without pivoting, in-place on A (n×n numpy fp32 array).

    Same block structure as sgetrf_nopiv_ttnn, offloaded to a CUDA GPU via
    PyTorch instead of a Tenstorrent Tensix via TTNN.

    Per block column j:
      Panel factorization (_sgetrf2_nopiv) : host, fp32   — serial recursive, (N-j)×32
      L11_inv                              : host, fp32   — invert the 32×32 pivot block
      Block-row strsm as matmul            : DEVICE, fp32 — U12 = L11_inv @ A12
      Trailing sgemm                       : DEVICE, fp32 — A22 -= A21 @ U12 (dominant O(N³))

    Unlike the TTNN path, CUDA fp32 matmul handles both operations natively
    in the same dtype, so there's no need for a bf16 round-trip of U12 —
    strsm and sgemm both stay fp32 and U12 can go straight from the strsm
    output into the sgemm input without leaving the device.

    Returns a dict of accumulated wall-clock seconds per phase.
    Note: CUDA kernel launches are async; device compute time that overlaps
    with subsequent host work surfaces in the d2h measurement (.cpu() blocks
    until the device drains). The strsm/sgemm buckets therefore capture
    dispatch latency only — d2h absorbs the remaining device execution time.
    """
    tm = dict(panel=0.0, inv=0.0, h2d=0.0, strsm=0.0, sgemm=0.0, d2h=0.0)
    _t = time.perf_counter

    n = A.shape[0]
    for j in range(0, n, NB):
        jb        = min(NB, n - j)
        remaining = max(0, n - j - jb)

        # ── panel factorization (host) ────────────────────────────────────
        t0 = _t(); _sgetrf2_nopiv(A[j:, j:j+jb]); tm["panel"] += _t() - t0

        if remaining == 0:
            break

        # ── invert pivot block on host (32×32, negligible cost) ──────────
        t0 = _t()
        L11 = np.tril(A[j:j+jb, j:j+jb].copy()).astype(np.float32)
        np.fill_diagonal(L11, 1.0)
        L11_inv = scipy.linalg.inv(L11)
        tm["inv"] += _t() - t0

        # ── host → device transfers (fp32 throughout) ─────────────────────
        t0 = _t()
        L11_inv_t = _to_cuda(L11_inv,              device)
        A12_t     = _to_cuda(A[j:j+jb, j+jb:],     device)
        A21_t     = _to_cuda(A[j+jb:, j:j+jb],     device)
        A22_t     = _to_cuda(A[j+jb:, j+jb:],      device)
        tm["h2d"] += _t() - t0

        # ── block-row strsm (device, fp32): U12 = L11_inv @ A12 ──────────
        t0 = _t(); U12_t = L11_inv_t @ A12_t; tm["strsm"] += _t() - t0

        # ── trailing sgemm (device, fp32): A22 -= A21 @ U12 ───────────────
        t0 = _t(); A22_t = A22_t - A21_t @ U12_t; tm["sgemm"] += _t() - t0

        # ── device → host write-back (.cpu() blocks until device drains) ──
        t0 = _t()
        A[j:j+jb, j+jb:] = _from_cuda(U12_t)
        A[j+jb:,  j+jb:] = _from_cuda(A22_t)
        tm["d2h"] += _t() - t0

    return tm


def _to_tt(arr: np.ndarray, device, dtype=None) -> "ttnn.Tensor":
    """numpy → contiguous torch → TTNN tile-layout tensor on device.

    dtype controls on-device storage precision; defaults to ttnn.float32.
    None as default avoids evaluating ttnn.float32 at module load time,
    since ttnn is imported lazily inside main().
    """
    t = torch.from_numpy(np.ascontiguousarray(arr))
    return ttnn.from_torch(
        t,
        dtype=dtype if dtype is not None else ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

def _from_tt(tt: "ttnn.Tensor") -> np.ndarray:
    """TTNN tensor → numpy (fp32). Casts to float32 first — numpy has no bfloat16."""
    return ttnn.to_torch(tt).to(torch.float32).numpy()


def sgetrf_nopiv_ttnn(A: np.ndarray, device) -> dict[str, float]:
    """
    Blocked LU without pivoting, in-place on A (n×n numpy fp32 array).

    Per block column j:
      Panel factorization (_sgetrf2_nopiv) : host, fp32   — serial recursive, (N-j)×32
      L11_inv                              : host, fp32   — invert the 32×32 pivot block
      Block-row strsm as matmul            : DEVICE, fp32 — U12 = L11_inv @ A12  (32×32 @ 32×(N-j-32))
      Trailing sgemm                       : DEVICE, bf16 — A22 -= A21 @ U12     (dominant O(N³))

    strsm stays fp32 because U12 is written back into A and seeds the next
    block column's panel factorization — precision lost there compounds
    across columns, not just within one trailing update. sgemm's operands
    (A21, A22) go bf16 for throughput, so a bf16 copy of U12 is needed to
    feed it; that copy is made with ttnn.typecast, which converts dtype
    on-device (no PCIe hop) — confirmed against the installed ttnn build
    via test_ttnn_typecast.py (numerically identical to a host round-trip,
    just without the transfer). The original fp32 U12_tt is untouched by
    the cast and remains what's written back to A.

    Returns a dict of accumulated wall-clock seconds per phase.
    Note: TTNN dispatch may be async; device compute time that overlaps with
    subsequent host work surfaces in the d2h measurement (to_torch blocks until
    the device drains). The strsm/cast/sgemm buckets therefore capture dispatch
    latency only — d2h absorbs the remaining device execution time.
    """
    tm = dict(panel=0.0, inv=0.0, h2d=0.0, strsm=0.0, cast=0.0, sgemm=0.0, d2h=0.0)
    _t = time.perf_counter   # local alias to shorten call sites

    n = A.shape[0]
    for j in range(0, n, NB):
        jb        = min(NB, n - j)
        remaining = max(0, n - j - jb)

        # ── panel factorization (host) ────────────────────────────────────
        t0 = _t(); _sgetrf2_nopiv(A[j:, j:j+jb]); tm["panel"] += _t() - t0

        if remaining == 0:
            break

        # ── invert pivot block on host (32×32, negligible cost) ──────────
        t0 = _t()
        L11 = np.tril(A[j:j+jb, j:j+jb].copy()).astype(np.float32)
        np.fill_diagonal(L11, 1.0)
        L11_inv = scipy.linalg.inv(L11)
        tm["inv"] += _t() - t0

        # ── host → device transfers (strsm operands stay fp32) ───────────
        t0 = _t()
        L11_inv_tt = _to_tt(L11_inv,           device)   # fp32 (default)
        A12_tt     = _to_tt(A[j:j+jb, j+jb:], device)   # fp32 (default)
        A21_tt     = _to_tt(A[j+jb:, j:j+jb], device, dtype=ttnn.bfloat16)
        A22_tt     = _to_tt(A[j+jb:, j+jb:],  device, dtype=ttnn.bfloat16)
        tm["h2d"] += _t() - t0

        # ── block-row strsm (device, fp32): U12 = L11_inv @ A12 ──────────
        t0 = _t(); U12_tt = ttnn.matmul(L11_inv_tt, A12_tt); tm["strsm"] += _t() - t0

        # ── on-device cast: bf16 copy of U12 for sgemm only, no PCIe hop.
        #    fp32 U12_tt is untouched and stays authoritative for the
        #    write-back below ─────────────────────────────────────────────
        t0 = _t()
        U12_bf16_tt = ttnn.typecast(U12_tt, ttnn.bfloat16)
        tm["cast"] += _t() - t0

        # ── trailing sgemm (device, bf16): A22 -= A21 @ U12 ───────────────
        t0 = _t()
        update  = ttnn.matmul(A21_tt, U12_bf16_tt)
        A22_out = ttnn.subtract(A22_tt, update)
        tm["sgemm"] += _t() - t0

        # ── device → host write-back (blocks until device drains) ─────────
        t0 = _t()
        A[j:j+jb, j+jb:] = _from_tt(U12_tt).astype(np.float32)   # fp32, unchanged precision
        A[j+jb:,  j+jb:] = _from_tt(A22_out).astype(np.float32)
        tm["d2h"] += _t() - t0

    return tm


# ─── GMRES iterative refinement: host, fp64 ──────────────────────────────────

def _rotmat(a: float, b: float):
    """Givens rotation: returns (c, s) such that [c s; -s c][a; b] = [r; 0]."""
    if b == 0.0:
        return 1.0, 0.0
    elif abs(b) > abs(a):
        t = a / b
        s = 1.0 / np.sqrt(1.0 + t * t)
        return t * s, s
    else:
        t = b / a
        c = 1.0 / np.sqrt(1.0 + t * t)
        return c, t * c


def _precond(LU: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply the fp32-LU preconditioner: solve L*(U*y) = v in fp64."""
    y = scipy.linalg.solve_triangular(LU, v, lower=True,  unit_diagonal=True)
    y = scipy.linalg.solve_triangular(LU, y, lower=False, unit_diagonal=False)
    return y


def gmres(
    n: int,
    A: np.ndarray,
    x: np.ndarray,
    b: np.ndarray,
    LU: np.ndarray,
    restart: int,
    max_it: int,
    tol: float,
) -> dict:
    """
    FGMRES (host, fp64).

    Parameters match the C call:
      gmres(n, A, lda, x, b, LU, ldlu, restart=max_it_cmd, max_it=1, tol)
    i.e. one outer restart cycle with up to `restart` inner Krylov steps.

    Gram-Schmidt orthogonalisation uses numpy vectorised dot/axpy — each
    H[k,i] = dot(w, V[:,k]) and w -= H[k,i]*V[:,k] are O(N) numpy calls,
    not Python scalar loops.

    Returns {"outer": int, "inner": int, "converged": bool,
             "max_outer": int, "max_inner": int, "residual_initial": float,
             "iters": list[tuple[int, float]]}.
    "iters" holds the (iteration, estimated_residual) trace printed during
    the inner Krylov loop, for callers that want it without reparsing stdout.
    """
    m      = min(restart, n)
    norm_b = np.linalg.norm(b) or 1.0
    eps    = np.finfo(np.float64).eps / 2

    # Iteration counters returned to the caller.
    outer_done = 0
    inner_done = 0
    converged  = False
    iters: list[tuple[int, float]] = []

    r      = _precond(LU, b - A @ x)
    error  = np.linalg.norm(r) / norm_b
    residual_initial = error
    print(f"Residual norm at beginning of GMRES: {error:.6e}")
    if error < tol:
        converged = True
        return dict(
            outer=0, inner=0, converged=True, max_outer=max_it, max_inner=m,
            residual_initial=residual_initial, iters=iters,
        )

    for _outer in range(max_it):
        outer_done = _outer + 1
        if _outer > 0:
            r = _precond(LU, b - A @ x)

        norm_r  = np.linalg.norm(r)
        V       = np.zeros((n, m + 1), dtype=np.float64)
        H       = np.zeros((m + 1, m), dtype=np.float64)
        cs      = np.zeros(m,          dtype=np.float64)
        sn      = np.zeros(m,          dtype=np.float64)
        s       = np.zeros(m + 1,      dtype=np.float64)
        V[:,0]  = r / norm_r
        s[0]    = norm_r
        updated = False

        for i in range(m):
            inner_done = i + 1   # track last completed inner step

            # Arnoldi step: w = M^{-1} A v_i
            w = _precond(LU, A @ V[:, i])

            # Gram-Schmidt orthogonalisation against V[:,0..i]
            for k in range(i + 1):
                H[k, i]  = np.dot(w, V[:, k])   # O(N) numpy dot
                w        -= H[k, i] * V[:, k]    # O(N) numpy axpy
            H[i+1, i]  = np.linalg.norm(w)
            V[:, i+1]  = w / H[i+1, i]

            # Apply previous Givens rotations to H[:, i]
            for k in range(i):
                tmp        =  cs[k] * H[k,   i] + sn[k] * H[k+1, i]
                H[k+1, i]  = -sn[k] * H[k,   i] + cs[k] * H[k+1, i]
                H[k,   i]  = tmp

            # Compute and apply i-th Givens rotation
            cs[i], sn[i] = _rotmat(H[i, i], H[i+1, i])
            s[i+1]       = -sn[i] * s[i]
            s[i]         =  cs[i] * s[i]
            H[i,   i]    =  cs[i] * H[i, i] + sn[i] * H[i+1, i]
            H[i+1, i]    = 0.0

            gmres_err = abs(s[i+1]) / norm_b
            iters.append((i + 1, gmres_err))
            print(f"  GMRES iter {i+1:3d}: estimated residual = {gmres_err:.6e}")

            if gmres_err <= tol:
                # H[0:i+1, 0:i+1] is upper-triangular after all Givens applied.
                y     = scipy.linalg.solve_triangular(H[:i+1, :i+1], s[:i+1])
                x_new = x + V[:, :i+1] @ y

                # HPL-AI scaled residual check (threshold = 16)
                res_new = b - A @ x_new
                err_hpl = (
                    np.max(np.abs(res_new))
                    / (
                        (np.max(np.sum(np.abs(A), axis=1)) * np.max(np.abs(x_new))
                         + np.max(np.abs(b)))
                        * n * eps
                    )
                )
                if err_hpl <= 16.0:
                    x[:]    = x_new
                    updated = True
                    break
                # HPL-AI check failed — keep building the Krylov space.

        if updated:
            # HPL-AI check already passed inside the inner loop — that is the
            # authoritative convergence criterion.  The outer residual recompute
            # below may show a slightly larger value than the GMRES estimate due
            # to finite-precision arithmetic, so we must not rely on it alone.
            converged = True
        else:
            y  = scipy.linalg.solve_triangular(H[:m, :m], s[:m])
            x += V[:, :m] @ y

        r = _precond(LU, b - A @ x)
        if converged or np.linalg.norm(r) / norm_b <= tol:
            break

    return dict(
        outer=outer_done, inner=inner_done,
        converged=converged, max_outer=max_it, max_inner=m,
        residual_initial=residual_initial, iters=iters,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def _load_raw_f64(path: str, expected_size: int | None = None) -> np.ndarray:
    """Load a headerless raw binary file of native-endian float64 values."""
    v = np.fromfile(path, dtype=np.float64)
    if expected_size is not None and v.size != expected_size:
        raise ValueError(
            f"{path}: expected {expected_size} float64 values, got {v.size}"
        )
    return v


def _fmt_metric(v) -> str:
    """Render one metric value for the machine-readable block.

    Floats use %.10g (fixed or scientific, whichever is shorter) so every
    value round-trips through float() without depending on column width or
    trailing-zero padding. Bools render as 1/0, matching the rest of the
    key=value block being plain numbers.
    """
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:.10g}"
    return str(v)


def _write_machine_readable(metrics: dict, path: str) -> None:
    """Write metrics as one `key=value` pair per line to `path`.

    Written in addition to (not instead of) the human-readable report on
    stdout — same run, same numbers, just also available in a file a script
    can load without touching stdout or reparsing the pretty formatting.
    File is plain `key=value` lines only, nothing else — trivially parsed
    with e.g. `dict(line.strip().split("=", 1) for line in open(path))`.
    """
    with open(path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}={_fmt_metric(v)}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hpl-ai.py",
        description="HPL-AI mixed-precision benchmark (Python, CPU/GPU/TTNN backend).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  hpl-ai.py                               run with defaults (N=100, MAX_IT=50, CPU)
  hpl-ai.py 500 40                        N=500, MAX_IT=40, matrix/vector generated
  hpl-ai.py --device gpu 500 40           same, LU factorization offloaded to CUDA
  hpl-ai.py --device ttnn 500 40          same, LU factorization offloaded to TTNN
  hpl-ai.py --input-a A.bin               N inferred from A.bin, b generated
  hpl-ai.py --input-a A.bin --input-b b.bin
                                           both A and b loaded from file

--input-a/--input-b files are headerless raw binary: contiguous native-endian
float64 values, A in row-major (C) order. This is the same layout a CUDA host
array (double A[N*N]) has in memory, so files can be produced/consumed with a
plain fwrite/fread on either side, with no format translation.""",
    )
    parser.add_argument(
        "n", nargs="?", type=int, default=100, metavar="N",
        help="problem size (default: 100). Ignored for A's size when "
             "--input-a is given, since N is then inferred from the file.",
    )
    parser.add_argument(
        "max_it", nargs="?", type=int, default=50, metavar="MAX_IT",
        help="GMRES restart length (default: 50), capped at N-1.",
    )
    parser.add_argument(
        "--device", choices=["cpu", "gpu", "ttnn"], default="cpu",
        help="backend for the LU factorization (default: cpu). "
             "'cpu' runs entirely on host (NumPy/SciPy, fp32), no accelerator "
             "required. 'gpu' offloads block-row strsm and the trailing "
             "sgemm to a CUDA GPU via PyTorch, fp32 (requires torch built "
             "with CUDA support). 'ttnn' offloads the same two ops to a "
             "Tenstorrent Tensix via TTNN, fp32/bf16 (requires torch/ttnn "
             "and a device at device_id=0).",
    )
    parser.add_argument(
        "--input-a", type=str, default=None, metavar="FILE",
        help="raw binary file: N*N float64 values, row-major. "
             "Overrides matgen(); N is inferred from the file size.",
    )
    parser.add_argument(
        "--input-b", type=str, default=None, metavar="FILE",
        help="raw binary file: N float64 values (requires --input-a). "
             "Overrides vecgen(); if omitted, b is still generated so you "
             "can pair a fixed A with a generated b.",
    )
    parser.add_argument(
        "--metrics-file", type=str, default="hpl-ai-metrics.txt", metavar="FILE",
        help="path to write machine-readable key=value metrics to "
             "(default: hpl-ai-metrics.txt in the current directory). "
             "Overwritten each run; not printed to stdout.",
    )
    args = parser.parse_args()

    if args.input_b and not args.input_a:
        parser.error("--input-b requires --input-a")

    return args


def main() -> None:
    # Declared once, unconditionally: CPython rejects a `global` statement
    # for a name that's already been *used* earlier in the function body,
    # even when that use sits in a different (mutually exclusive) branch.
    global torch, ttnn

    args  = _parse_args()
    iseed = 1

    print("=" * 78)
    print("              HPL-AI Mixed-Precision Benchmark  [Python]")
    print("=" * 78)
    backend_label = {
        "cpu":  "CPU (NumPy/SciPy)",
        "gpu":  "GPU (PyTorch/CUDA)",
        "ttnn": "TTNN (Tenstorrent Tensix)",
    }[args.device]
    print(f"Backend: {backend_label}")

    device = None
    if args.device == "gpu":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device gpu requires a CUDA-capable GPU, but "
                "torch.cuda.is_available() is False."
            )
        device = torch.device("cuda")
    elif args.device == "ttnn":
        import torch
        import ttnn

        device = ttnn.open_device(device_id=0)

    # ── Load A and b from file, or generate (fp64) ───────────────────────────
    if args.input_a:
        flat = _load_raw_f64(args.input_a)
        n    = int(round(flat.size ** 0.5))
        if n * n != flat.size:
            raise ValueError(
                f"{args.input_a}: {flat.size} values is not a perfect square"
            )
        A = flat.reshape(n, n)
        b = _load_raw_f64(args.input_b, expected_size=n) if args.input_b \
            else vecgen(n, iseed + 1)
    else:
        n = args.n
        A = matgen(n, iseed)
        b = vecgen(n, iseed + 1)

    max_it = min(args.max_it, n - 1)

    # ── fp64 → fp32 ──────────────────────────────────────────────────────────
    t0    = time.perf_counter()
    sA    = A.astype(np.float32)
    sb    = b.astype(np.float32)
    t_c1  = time.perf_counter() - t0
    print(f"Time: conversion to single      {1e3 * t_c1:12.3f} ms")

    # ── LU factorization (fp32; host-only on cpu, strsm+sgemm offloaded on gpu/ttnn) ─
    t0 = time.perf_counter()
    if args.device == "gpu":
        lu_tm = sgetrf_nopiv_gpu(sA, device)
    elif args.device == "ttnn":
        lu_tm = sgetrf_nopiv_ttnn(sA, device)
    else:
        lu_tm = sgetrf_nopiv_cpu(sA)
    t_factor = time.perf_counter() - t0

    lu_sum = sum(lu_tm.values())   # should ≈ t_factor

    _W = 12   # column width for ms values
    def _lu_row(label, key, connector="├─"):
        ms  = 1e3 * lu_tm[key]
        pct = 100.0 * lu_tm[key] / lu_sum if lu_sum > 0 else 0.0
        print(f"  {connector} {label:<38s} {ms:{_W}.3f} ms  ({pct:5.1f}%)")

    print(f"Time: LU factorization          {1e3 * t_factor:{_W}.3f} ms")
    if args.device == "ttnn":
        _lu_row("panel  _sgetrf2_nopiv  (host)",  "panel")
        _lu_row("inv    L11⁻¹ 32×32     (host)",  "inv")
        _lu_row("h2d    host → device   (PCIe)",  "h2d")
        _lu_row("strsm  U12=L11⁻¹·A12  (TTNN)",   "strsm")
        _lu_row("cast   U12 fp32→bf16  (TTNN)",   "cast")
        _lu_row("sgemm  A22-=A21·U12   (TTNN)",   "sgemm")
        _lu_row("d2h    device → host   (PCIe)",  "d2h",  connector="└─")
    elif args.device == "gpu":
        _lu_row("panel  _sgetrf2_nopiv  (host)",  "panel")
        _lu_row("inv    L11⁻¹ 32×32     (host)",  "inv")
        _lu_row("h2d    host → device   (PCIe)",  "h2d")
        _lu_row("strsm  U12=L11⁻¹·A12  (CUDA)",   "strsm")
        _lu_row("sgemm  A22-=A21·U12   (CUDA)",   "sgemm")
        _lu_row("d2h    device → host   (PCIe)",  "d2h",  connector="└─")
    else:
        _lu_row("panel  sgetrf2_nopiv  (host)",  "panel")
        _lu_row("strsm  U12=L11⁻¹·A12 (host)",   "strsm")
        _lu_row("sgemm  A22-=A21·U12  (host)",   "sgemm", connector="└─")

    # ── Initial triangular solve (fp32, host) ─────────────────────────────────
    t0      = time.perf_counter()
    sb      = scipy.linalg.solve_triangular(sA, sb, lower=True,  unit_diagonal=True)
    sb      = scipy.linalg.solve_triangular(sA, sb, lower=False, unit_diagonal=False)
    t_solve = time.perf_counter() - t0
    print(f"Time: triangular solve          {1e3 * t_solve:12.3f} ms")

    # ── fp32 → fp64 ──────────────────────────────────────────────────────────
    t0   = time.perf_counter()
    LU   = sA.astype(np.float64)
    x    = sb.astype(np.float64)
    t_c2 = time.perf_counter() - t0
    print(f"Time: conversion to double      {1e3 * t_c2:12.3f} ms")

    # ── GMRES iterative refinement (fp64, host) ───────────────────────────────
    t0      = time.perf_counter()
    eps     = np.finfo(np.float64).eps / 2
    tol     = eps / (n / 4.0)
    # Call signature mirrors C: gmres(..., restart=max_it, max_it_outer=1, tol)
    gm      = gmres(n, A, x, b, LU, max_it, 1, tol)
    t_gmres = time.perf_counter() - t0
    print(f"Time: GMRES                     {1e3 * t_gmres:12.3f} ms")
    if gm["converged"]:
        print(f"  GMRES converged in {gm['inner']} / {gm['max_inner']} iterations")
    else:
        print(f"  GMRES did NOT converge — reached limit ({gm['inner']} / {gm['max_inner']} iterations)")

    t_total = t_c1 + t_factor + t_solve + t_c2 + t_gmres
    ops     = (2.0 / 3.0) * n**3 + 1.5 * n**2
    print(f"Total time                      {1e3 * t_total:12.3f} ms")
    print(f"Effective GFLOPs                {1e-9 * ops / t_total:12.6f}")

    # ── Final scaled-residual check ───────────────────────────────────────────
    norm_A   = np.max(np.sum(np.abs(A), axis=1))   # ||A||_inf
    norm_x   = np.max(np.abs(x))
    norm_b   = np.max(np.abs(b))
    residual = b - A @ x
    norm_r   = np.max(np.abs(residual))
    error    = norm_r / ((norm_A * norm_x + norm_b) * n * eps)

    print()
    print("||Ax-b||_oo / ( eps * ( ||x||_oo * ||A||_oo + ||b||_oo ) * N )")
    print(f"eps = {eps:.6e}")
    passed = bool(error < 16.0)
    print(f"Scaled residual = {error:.6f}  ...", "PASSED" if passed else "FAILED")

    # ── Machine-readable summary (same numbers, key=value form) ──────────────
    metrics = {
        "device": args.device,
        "n": n,
        "max_it": max_it,
        "nb": NB,
        "time_convert_single_ms": 1e3 * t_c1,
        "time_lu_factorization_ms": 1e3 * t_factor,
    }
    for key, secs in lu_tm.items():
        metrics[f"lu_{key}_ms"] = 1e3 * secs
        metrics[f"lu_{key}_pct"] = 100.0 * secs / lu_sum if lu_sum > 0 else 0.0
    metrics.update({
        "time_triangular_solve_ms": 1e3 * t_solve,
        "time_convert_double_ms": 1e3 * t_c2,
        "gmres_residual_initial": gm["residual_initial"],
    })
    for i, res in gm["iters"]:
        metrics[f"gmres_iter_{i}_residual"] = res
    metrics.update({
        "time_gmres_ms": 1e3 * t_gmres,
        "gmres_converged": gm["converged"],
        "gmres_outer_iterations": gm["outer"],
        "gmres_max_outer_iterations": gm["max_outer"],
        "gmres_inner_iterations": gm["inner"],
        "gmres_max_inner_iterations": gm["max_inner"],
        "time_total_ms": 1e3 * t_total,
        "gflops": 1e-9 * ops / t_total,
        "eps": eps,
        "scaled_residual": error,
        "scaled_residual_passed": passed,
    })
    _write_machine_readable(metrics, args.metrics_file)
    print(f"\nMachine-readable metrics written to: {args.metrics_file}")

    if args.device == "ttnn":
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
