"""
tuned_matmul_bench.py — Isolates whether the huge TT vs GPU matmul/trsm gap
seen in the LU solver benchmark is a *config problem in that solver's ttnn
calls* (small/default core grid, unspecified program_config) or a more
fundamental throughput ceiling. Runs a deliberately, explicitly tuned
ttnn.matmul at the same sizes the solver used (1024/2048/4096/10240/20480 by
default), plus H2D/D2H bandwidth at the same sizes, so both can be compared
against each device's theoretical peak (see the efficiency-normalization
notes at the bottom of this file).

What "tuned" means here, concretely:
  - core_grid explicitly set to the device's FULL compute_with_storage_grid_size()
    (override with --core-grid-x/--core-grid-y if that query isn't available in
    your ttnn version, or if you want to test a specific smaller grid on purpose).
  - fp32_dest_acc_en=True, math_approx_mode=False (max precision path, see
    README_tenstorrent_cache_findings.md / earlier fp32-accuracy scripts for why).
  - every MathFidelity level tested (LoFi/HiFi2/HiFi3/HiFi4), since peak achievable
    TFLOPS differs ~4x across them (see matrix_engine.md: LoFi=4, HiFi2=2,
    HiFi3=1.33, HiFi4=1 TFLOPS *per matrix engine* at 1GHz -- multiply by the
    number of Tensix cores in your grid to get the config's theoretical peak).
  - a "default" run (program_config=None, no explicit core_grid) alongside the
    tuned ones, so you can directly see how much the solver's likely-unconfigured
    call was leaving on the table.

Usage:
    python tuned_matmul_bench.py --sizes 1024,2048,4096,10240,20480 --csv tt_tuned.csv
    python tuned_matmul_bench.py --sizes 20480 --core-grid-x 8 --core-grid-y 8 --csv probe.csv

Output CSV columns: size, fidelity, core_grid, warm_ms, achieved_gflops,
h2d_gbps, d2h_gbps. No peak/percentage columns are written here -- this
script only measures what actually happened. Compute % of peak yourself
once you know your board's real numbers (see the normalization notes at
the bottom of this file, and the companion tuned_matmul_bench.cu for the
CUDA side).
"""

import argparse
import csv
import sys
import time


def make_sync_fn(ttnn, device):
    """ttnn's blocking-sync entry point has moved across versions; probe once
    at startup and fail loudly rather than silently falling back to a no-op
    (a silent no-op here is exactly how the original GFLOPS bug happened)."""
    for candidate in ("synchronize_device", "SynchronizeDevice"):
        fn = getattr(ttnn, candidate, None)
        if fn is not None:
            print(f"[sync] using ttnn.{candidate}(device)", file=sys.stderr)
            def sync():
                fn(device)
            return sync
    if hasattr(device, "synchronize"):
        print("[sync] using device.synchronize()", file=sys.stderr)
        def sync():
            device.synchronize()
        return sync
    print("[Error] Could not find a device-synchronize function on this ttnn build "
          "(tried ttnn.synchronize_device, ttnn.SynchronizeDevice, device.synchronize). "
          "Timed results would be meaningless without it -- check your ttnn version's "
          "API and wire it in before trusting any GFLOPS number from this script.",
          file=sys.stderr)
    sys.exit(1)


# Per-matrix-engine TFLOPS at 1GHz, from tt-metal's matrix_engine.md. Multiply
# by the number of active Tensix cores (grid_x * grid_y) to get that config's
# theoretical ceiling. Used only for the plausibility check below and for the
# pct_of_peak column -- NOT hardcoded elsewhere, since this is exactly the
# "peak FLOPS normalization" the earlier methodology discussion asked for.
PEAK_TFLOPS_PER_ENGINE = {"LoFi": 4.0, "HiFi2": 2.0, "HiFi3": 1.33, "HiFi4": 1.0}


def peak_gflops(fid_name, grid_x, grid_y):
    per_engine = PEAK_TFLOPS_PER_ENGINE.get(fid_name)
    if per_engine is None:  # e.g. "default(unconfigured)" -- unknown fidelity/grid
        return None
    return per_engine * grid_x * grid_y * 1000.0


def get_core_grid(ttnn, device, override_x, override_y):
    if override_x is not None and override_y is not None:
        return override_x, override_y
    try:
        grid = device.compute_with_storage_grid_size()
        return grid.x, grid.y
    except Exception as e:
        print(f"[Warning] Could not query device.compute_with_storage_grid_size(): {e}\n"
              f"Pass --core-grid-x/--core-grid-y explicitly.", file=sys.stderr)
        sys.exit(1)


def make_compute_kernel_config(ttnn, fidelity):
    cfg_cls = getattr(ttnn, "BlackholeComputeKernelConfig", None) or ttnn.WormholeComputeKernelConfig
    return cfg_cls(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def time_call(fn, warmups=1):
    for _ in range(warmups):
        fn()
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0  # ms


def bench_size(ttnn, torch, device, size, fidelities, grid_x, grid_y, rows, sync):
    import numpy as np

    m = k = n = size
    rng = np.random.default_rng(0)
    a_np = rng.uniform(-1, 1, size=(m, k)).astype(np.float32)
    b_np = rng.uniform(-1, 1, size=(k, n)).astype(np.float32)
    a_t = torch.tensor(a_np, dtype=torch.float32)
    b_t = torch.tensor(b_np, dtype=torch.float32)

    # ---- H2D bandwidth (both operand matrices) ----
    # NOTE: ttnn dispatches ops asynchronously -- from_torch()/matmul() enqueue
    # work and return once it's queued, not once the device has finished. Every
    # timed region below MUST end with ttnn.synchronize_device(device) or a
    # blocking readback (ttnn.to_torch), or you measure host-side dispatch
    # latency instead of real device time. (This is exactly the bug that
    # produced physically-impossible >100,000 GFLOPS numbers in the first
    # version of this script -- warm_ms stayed ~flat across a 20x size range
    # instead of scaling with N^3, because deallocate() alone does not block.)
    def do_h2d():
        ad = ttnn.from_torch(a_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        bd = ttnn.from_torch(b_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        sync()
        return ad, bd

    # warm up the from_torch path once, then time a fresh (still "warm", same
    # shape/dtype/program) call -- this is about steady-state transfer speed,
    # not JIT/cache effects, which earlier scripts already covered separately.
    a_dev, b_dev = do_h2d()
    ttnn.deallocate(a_dev)
    ttnn.deallocate(b_dev)
    t0 = time.perf_counter()
    a_dev, b_dev = do_h2d()
    t1 = time.perf_counter()
    h2d_bytes = (a_np.nbytes + b_np.nbytes)
    h2d_gbps = h2d_bytes / ((t1 - t0)) / 1e9

    for fid_name, fid in fidelities:
        cfg = make_compute_kernel_config(ttnn, fid)
        core_grid = ttnn.CoreGrid(y=grid_y, x=grid_x)
        flops = 2.0 * m * k * n

        # synchronize_device() before stopping the timer -- confirmed correct
        # against theoretical peak (see pct_of_peak below); the earlier
        # to_torch()-readback cross-check was a one-time diagnostic and has
        # been removed now that sync_only is validated.
        def run():
            out = ttnn.matmul(a_dev, b_dev, dtype=ttnn.float32, compute_kernel_config=cfg, core_grid=core_grid)
            sync()
            return out

        for _ in range(2):
            ttnn.deallocate(run())
        t0 = time.perf_counter()
        out_a = run()
        t1 = time.perf_counter()
        warm_ms = (t1 - t0) * 1000.0
        ttnn.deallocate(out_a)
        achieved_gflops = flops / (warm_ms / 1000.0) / 1e9

        peak = peak_gflops(fid_name, grid_x, grid_y)
        pct_of_peak = achieved_gflops / peak * 100.0 if peak else None
        # 50% headroom over the theoretical ceiling as the "still broken" bar --
        # real efficiency should land under 100% of peak, not orders of magnitude
        # over it like the original (unsynchronized) numbers did.
        if peak and achieved_gflops > peak * 1.5:
            print(f"  [Warning] size={size} fidelity={fid_name}: sync-only={achieved_gflops:.1f} GFLOPS "
                  f"exceeds 150% of this config's theoretical peak ({peak:.1f} GFLOPS) -- still suspect.",
                  file=sys.stderr)

        rows.append({
            "size": size, "fidelity": fid_name, "core_grid": f"{grid_x}x{grid_y}",
            "warm_ms": warm_ms, "achieved_gflops": achieved_gflops,
            "peak_gflops": peak, "pct_of_peak": pct_of_peak,
            "h2d_gbps": h2d_gbps, "d2h_gbps": None,
        })
        pct_str = f"{pct_of_peak:5.1f}%" if pct_of_peak is not None else "  n/a"
        print(f"  size={size:6d} fidelity={fid_name:6s} core_grid={grid_x}x{grid_y}  "
              f"warm={warm_ms:9.3f}ms  {achieved_gflops:12.2f} GFLOPS ({pct_str} of peak)")

    # ---- default (unconfigured) run, for direct A/B against the tuned ones ----
    def run_default():
        out = ttnn.matmul(a_dev, b_dev, dtype=ttnn.float32)
        sync()
        ttnn.deallocate(out)

    warm_ms_default = time_call(run_default, warmups=2)
    flops = 2.0 * m * k * n
    achieved_gflops_default = flops / (warm_ms_default / 1000.0) / 1e9
    rows.append({
        "size": size, "fidelity": "default(unconfigured)", "core_grid": "auto",
        "warm_ms": warm_ms_default, "achieved_gflops": achieved_gflops_default,
        "peak_gflops": None, "pct_of_peak": None,
        "h2d_gbps": h2d_gbps, "d2h_gbps": None,
    })
    print(f"  size={size:6d} fidelity=default(unconfigured) core_grid=auto     "
          f"warm={warm_ms_default:9.3f}ms  {achieved_gflops_default:9.2f} GFLOPS")

    # ---- D2H bandwidth (result of the last matmul) ----
    out_dev = ttnn.matmul(a_dev, b_dev, dtype=ttnn.float32)
    _ = ttnn.to_torch(out_dev)  # warm up
    t0 = time.perf_counter()
    out_np = ttnn.to_torch(out_dev)
    t1 = time.perf_counter()
    d2h_bytes = out_np.numpy().nbytes if hasattr(out_np, "numpy") else (m * n * 4)
    d2h_gbps = d2h_bytes / (t1 - t0) / 1e9
    for r in rows:
        if r["size"] == size and r["d2h_gbps"] is None:
            r["d2h_gbps"] = d2h_gbps
    print(f"  size={size:6d} H2D={h2d_gbps:7.2f} GB/s   D2H={d2h_gbps:7.2f} GB/s")

    ttnn.deallocate(a_dev)
    ttnn.deallocate(b_dev)
    ttnn.deallocate(out_dev)


def parse_sizes_file(path):
    """One size per line. Blank lines and lines starting with '#' are skipped.
    e.g.:
        1234
        4567
        7890
    is equivalent to --sizes 1234,4567,7890 ."""
    sizes = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                sizes.append(int(line))
            except ValueError:
                print(f"[Error] {path}:{lineno}: {line!r} is not an integer size.", file=sys.stderr)
                sys.exit(1)
    if not sizes:
        print(f"[Error] {path} contained no sizes.", file=sys.stderr)
        sys.exit(1)
    return sizes


def main():
    parser = argparse.ArgumentParser(description="Tuned ttnn matmul + H2D/D2H bandwidth benchmark")
    parser.add_argument("--sizes", type=str, default="1024,2048,4096,10240,20480",
                         help="Comma-separated problem sizes. Ignored if --sizes-file is given.")
    parser.add_argument("--sizes-file", type=str, default=None,
                         help="Path to a file with one problem size per line (# comments/blank lines "
                              "allowed). Takes priority over --sizes. A file containing:\n"
                              "  1234\n  4567\n  7890\nbehaves identically to --sizes 1234,4567,7890.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--core-grid-x", type=int, default=None)
    parser.add_argument("--core-grid-y", type=int, default=None)
    parser.add_argument("--csv", type=str, default="tuned_matmul_ttnn.csv")
    args = parser.parse_args()

    try:
        import torch
        import ttnn
    except ImportError as e:
        print(f"[Error] Failed to import torch/ttnn: {e}", file=sys.stderr)
        sys.exit(1)

    if args.sizes_file:
        sizes = parse_sizes_file(args.sizes_file)
    else:
        sizes = [int(x) for x in args.sizes.split(",")]
    fidelities = [
        ("LoFi", ttnn.MathFidelity.LoFi),
        ("HiFi2", ttnn.MathFidelity.HiFi2),
        ("HiFi3", ttnn.MathFidelity.HiFi3),
        ("HiFi4", ttnn.MathFidelity.HiFi4),
    ]

    device = ttnn.open_device(device_id=args.device_id)
    try:
        grid_x, grid_y = get_core_grid(ttnn, device, args.core_grid_x, args.core_grid_y)
        print(f"Using core grid: {grid_x}x{grid_y} (pass --core-grid-x/-y to override)")
        sync = make_sync_fn(ttnn, device)

        rows = []
        for size in sizes:
            print(f"\n=== size={size} ===")
            bench_size(ttnn, torch, device, size, fidelities, grid_x, grid_y, rows, sync)
    finally:
        ttnn.close_device(device)

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["size", "fidelity", "core_grid", "warm_ms",
                                                "achieved_gflops", "peak_gflops", "pct_of_peak",
                                                "h2d_gbps", "d2h_gbps"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Normalizing to "% of peak" (do this after you have achieved_gflops / h2d_gbps
# / d2h_gbps in the CSV -- this script deliberately does NOT hardcode your
# hardware's peak specs, since guessing wrong is worse than not guessing).
#
# FLOPS efficiency (compute kernel):
#   pct_of_peak = achieved_gflops / 1000 / peak_tflops_for_this_exact_config * 100
#
#   Finding peak_tflops for Blackhole at a given run:
#     peak_tflops = per_matrix_engine_tflops[fidelity] * num_tensix_cores_in_core_grid
#     per_matrix_engine_tflops (Wormhole/Blackhole matrix engine, from
#     tt-metal's matrix_engine.md, at 1GHz): LoFi=4, HiFi2=2, HiFi3=1.33, HiFi4=1
#     num_tensix_cores_in_core_grid = grid_x * grid_y (printed by this script)
#
#   Finding peak_tflops for the GPU side: look up the TF32 Tensor Core TFLOPS
#   (sparsity OFF -- this benchmark's GEMMs are dense) figure on the vendor's
#   datasheet/whitepaper for your EXACT GPU model. Do not mix this up with the
#   plain FP32 CUDA-core figure -- they can differ by ~8-10x on the same chip,
#   and which one applies depends on whether cuBLAS actually took the tensor-
#   core path (CUBLAS_COMPUTE_32F_FAST_TF32) or the plain path (CUBLAS_COMPUTE_32F)
#   for that call -- see tuned_matmul_bench.cu, which reports both separately.
#
# PCIe bandwidth efficiency (H2D/D2H):
#   pct_of_peak = achieved_gbps / theoretical_peak_gbps * 100
#
#   Finding theoretical_peak_gbps: identify the PCIe generation and negotiated
#   lane width actually in use for that device:
#     Linux:  lspci -vv -s <device_bdf> | grep -E "LnkCap|LnkSta"
#             (LnkSta shows what's actually negotiated, which is what matters --
#             a card capable of Gen5 x16 sitting in a Gen4 x8 slot is only
#             getting Gen4 x8 bandwidth in practice)
#     NVIDIA: nvidia-smi -q | grep -A5 "GPU Link Info" also gives this directly.
#   Then look up GB/s per lane per generation (single direction, real payload
#   throughput after 128b/130b or 128b/132b encoding overhead):
#     Gen3 ~= 0.985 GB/s/lane   Gen4 ~= 1.969 GB/s/lane
#     Gen5 ~= 3.938 GB/s/lane   Gen6 ~= 7.563 GB/s/lane (PAM4 + FEC overhead)
#   theoretical_peak_gbps = GB/s_per_lane[generation] * negotiated_lane_width
#
#   Note: even a well-implemented transfer path realistically lands around
#   70-90% of this raw theoretical number due to protocol/DMA-setup overhead,
#   so don't expect ~100% even from a "healthy" measurement -- compare against
#   that ~80% realistic band, not the raw theoretical ceiling.
# ---------------------------------------------------------------------------