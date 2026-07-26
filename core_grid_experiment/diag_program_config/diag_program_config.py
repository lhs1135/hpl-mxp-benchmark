"""
diag_program_config.py — Diagnoses WHY ttnn.matmul's efficiency swings wildly
(observed: ~90-100% of peak at some sizes, ~24% at others just 1024 apart,
non-monotonic in N -- see tuned_matmul_ttnn_scalability analysis) by manually
sweeping ttnn.MatmulMultiCoreReuseMultiCastProgramConfig instead of relying on
core_grid=... convenience auto-selection.

Rationale: single-variable checks (N mod 1024, N mod 32, tile-count mod
grid_x/grid_y) do NOT cleanly separate "good" from "bad" sizes. A rough
simulation of tt-metal's likely subblock-selection logic (largest
out_subblock_h * out_subblock_w <= 8 that evenly divides the per-core tile
counts) correlates only moderately (r~0.55) with measured efficiency -- not
conclusive on its own. This script tests the hypothesis directly: at a single
"bad" size, sweep real program_config combinations and see whether ANY of them
recovers high efficiency.

  - If some manual combo at a "bad" size (e.g. 20480) reaches ~80-90%+ of peak
    while core_grid=... auto-selection gets ~24%: the auto-config heuristic is
    choosing badly for that size, and the fix is to always pass an explicit
    program_config (never rely on core_grid alone) for sizes that matter.
  - If NO manual combo helps at that size: something other than subblock
    selection is the bottleneck at that N (e.g. a real memory/bandwidth wall),
    and this needs a different kind of investigation (device-side profiling).

Field names come from the official ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
docs (compute_with_storage_grid_size, in0_block_w, out_subblock_h, out_subblock_w,
per_core_M, per_core_N, transpose_mcast, fused_activation, fuse_batch). The exact
constructor keyword-argument acceptance can vary slightly by tt-metal version --
if construction fails, this script prints what IS available via dir() so you can
adjust field names quickly rather than guessing blind.

Usage:
    python diag_program_config.py --size 20480 --fidelity HiFi4 --csv diag_20480.csv
    python diag_program_config.py --size 30720 --fidelity HiFi4 --csv diag_30720.csv
    # then compare: does 20480 have ANY combo reaching close to 30720's ~100%?

CSV columns: size, in0_block_w, out_subblock_h, out_subblock_w, per_core_M,
per_core_N, warm_ms, achieved_gflops, pct_of_peak, status (ok / skipped:<reason>)
"""

import argparse
import csv
import math
import sys
import time


def divisors(n):
    ds = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            ds.append(i)
            if i != n // i:
                ds.append(n // i)
    return sorted(ds)


PEAK_TFLOPS_PER_ENGINE = {"LoFi": 4.0, "HiFi2": 2.0, "HiFi3": 1.33, "HiFi4": 1.0}


def make_sync_fn(ttnn, device):
    for candidate in ("synchronize_device", "SynchronizeDevice"):
        fn = getattr(ttnn, candidate, None)
        if fn is not None:
            def sync():
                fn(device)
            return sync
    if hasattr(device, "synchronize"):
        def sync():
            device.synchronize()
        return sync
    print("[Error] No device-synchronize function found -- see tuned_matmul_bench.py's "
          "make_sync_fn for the same check. Refusing to time anything without it.", file=sys.stderr)
    sys.exit(1)


def get_core_grid(device, override_x, override_y):
    if override_x is not None and override_y is not None:
        return override_x, override_y
    grid = device.compute_with_storage_grid_size()
    return grid.x, grid.y


def build_program_config(ttnn, grid_x, grid_y, in0_block_w, out_subblock_h, out_subblock_w,
                          per_core_M, per_core_N):
    """Try a couple of plausible constructor shapes; report available fields if
    all of them fail, rather than a bare, unhelpful traceback."""
    kwargs = dict(
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
    )
    attempts = []
    for grid_value in (
        getattr(ttnn, "CoreCoord", lambda x, y: (x, y))(grid_x, grid_y),
        (grid_x, grid_y),
    ):
        attempts.append(dict(kwargs, compute_with_storage_grid_size=grid_value))

    last_err = None
    for attempt_kwargs in attempts:
        try:
            return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(**attempt_kwargs)
        except Exception as e:
            last_err = e
    print(f"[Error] Could not construct MatmulMultiCoreReuseMultiCastProgramConfig on this "
          f"ttnn build: {last_err}\n"
          f"Available attributes: {[a for a in dir(ttnn.MatmulMultiCoreReuseMultiCastProgramConfig) if not a.startswith('_')]}\n"
          f"Adjust build_program_config()'s kwargs to match your ttnn version.", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep explicit MatmulMultiCoreReuseMultiCastProgramConfig at one size "
                    "to test whether manual config recovers efficiency lost to auto-selection.")
    parser.add_argument("--size", type=int, required=True, help="Square M=K=N problem size to test.")
    parser.add_argument("--fidelity", type=str, default="HiFi4", choices=list(PEAK_TFLOPS_PER_ENGINE))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--core-grid-x", type=int, default=None)
    parser.add_argument("--core-grid-y", type=int, default=None)
    parser.add_argument("--max-combos", type=int, default=40,
                         help="Cap on how many (in0_block_w, out_subblock_h, out_subblock_w) "
                              "combinations to try, to keep runtime bounded.")
    parser.add_argument("--csv", type=str, default="diag_program_config.csv")
    args = parser.parse_args()

    try:
        import torch
        import ttnn
    except ImportError as e:
        print(f"[Error] Failed to import torch/ttnn: {e}", file=sys.stderr)
        sys.exit(1)

    fieldnames = ["size", "in0_block_w", "out_subblock_h", "out_subblock_w", "per_core_M",
                  "per_core_N", "warm_ms", "achieved_gflops", "pct_of_peak", "status"]
    csv_file = open(args.csv, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    def write_row(row):
        writer.writerow(row)
        csv_file.flush()

    device = ttnn.open_device(device_id=args.device_id)
    try:
        grid_x, grid_y = get_core_grid(device, args.core_grid_x, args.core_grid_y)
        print(f"core grid: {grid_x}x{grid_y}")
        sync = make_sync_fn(ttnn, device)

        size = args.size
        m = k = n = size
        tiles = size // 32
        if size % 32 != 0:
            print(f"[Error] size={size} is not a multiple of 32 (the tile size) -- "
                  f"pick a multiple of 32 so tile counts are exact.", file=sys.stderr)
            sys.exit(1)

        import numpy as np
        rng = np.random.default_rng(0)
        a_np = rng.uniform(-1, 1, size=(m, k)).astype(np.float32)
        b_np = rng.uniform(-1, 1, size=(k, n)).astype(np.float32)
        a_dev = ttnn.from_torch(torch.from_numpy(a_np), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        b_dev = ttnn.from_torch(torch.from_numpy(b_np), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        sync()

        fid = getattr(ttnn.MathFidelity, args.fidelity)
        cfg_cls = getattr(ttnn, "BlackholeComputeKernelConfig", None) or ttnn.WormholeComputeKernelConfig
        compute_kernel_config = cfg_cls(math_fidelity=fid, math_approx_mode=False,
                                         fp32_dest_acc_en=True, packer_l1_acc=False)
        peak = PEAK_TFLOPS_PER_ENGINE[args.fidelity] * grid_x * grid_y * 1000.0
        flops = 2.0 * m * k * n

        def time_matmul(program_config):
            def run():
                out = ttnn.matmul(a_dev, b_dev, dtype=ttnn.float32, compute_kernel_config=compute_kernel_config,
                                   program_config=program_config)
                sync()
                return out
            for _ in range(2):
                ttnn.deallocate(run())
            t0 = time.perf_counter()
            out = run()
            t1 = time.perf_counter()
            ttnn.deallocate(out)
            return (t1 - t0) * 1000.0

        # ---- Reference row: today's auto-selection via core_grid=... alone ----
        core_grid = ttnn.CoreGrid(y=grid_y, x=grid_x)
        try:
            def run_auto():
                out = ttnn.matmul(a_dev, b_dev, dtype=ttnn.float32, compute_kernel_config=compute_kernel_config,
                                   core_grid=core_grid)
                sync()
                return out
            for _ in range(2):
                ttnn.deallocate(run_auto())
            t0 = time.perf_counter()
            out = run_auto()
            t1 = time.perf_counter()
            ttnn.deallocate(out)
            warm_ms = (t1 - t0) * 1000.0
            gflops = flops / (warm_ms / 1000.0) / 1e9
            write_row({"size": size, "in0_block_w": "auto", "out_subblock_h": "auto",
                       "out_subblock_w": "auto", "per_core_M": "auto", "per_core_N": "auto",
                       "warm_ms": warm_ms, "achieved_gflops": gflops,
                       "pct_of_peak": gflops / peak * 100.0, "status": "ok(core_grid auto-select)"})
            print(f"[baseline] core_grid auto-select: {warm_ms:.3f}ms  {gflops:.1f} GFLOPS "
                  f"({gflops/peak*100:.1f}% of peak)")
        except Exception as e:
            print(f"[Warning] baseline core_grid run failed: {e}", file=sys.stderr)

        # ---- Manual sweep ----
        per_core_M = math.ceil(tiles / grid_y)
        per_core_N = math.ceil(tiles / grid_x)
        k_tiles = tiles  # K == N == M here (square)

        h_divs = [d for d in divisors(per_core_M) if d <= per_core_M]
        w_divs = [d for d in divisors(per_core_N) if d <= per_core_N]
        subblock_combos = sorted(
            {(h, w) for h in h_divs for w in w_divs if h * w <= 8},
            key=lambda hw: -(hw[0] * hw[1])  # try the largest (most efficient-looking) subblocks first
        )
        # A handful of in0_block_w candidates: full K, and a few small divisors
        # (smaller blocks trade compute granularity for lower L1 footprint / more
        # multicast overhead -- exactly the kind of tradeoff worth sweeping).
        in0_candidates = sorted({d for d in divisors(k_tiles) if d <= k_tiles}, reverse=True)[:6]

        tried = 0
        print(f"per_core_M={per_core_M} per_core_N={per_core_N} k_tiles={k_tiles}  "
              f"{len(subblock_combos)} subblock combos x {len(in0_candidates)} in0_block_w candidates")
        for in0_block_w in in0_candidates:
            for out_subblock_h, out_subblock_w in subblock_combos:
                if tried >= args.max_combos:
                    print(f"[Info] Hit --max-combos={args.max_combos}, stopping sweep early.")
                    break
                tried += 1
                try:
                    pc = build_program_config(ttnn, grid_x, grid_y, in0_block_w, out_subblock_h,
                                               out_subblock_w, per_core_M, per_core_N)
                    warm_ms = time_matmul(pc)
                    gflops = flops / (warm_ms / 1000.0) / 1e9
                    pct = gflops / peak * 100.0
                    write_row({"size": size, "in0_block_w": in0_block_w, "out_subblock_h": out_subblock_h,
                               "out_subblock_w": out_subblock_w, "per_core_M": per_core_M,
                               "per_core_N": per_core_N, "warm_ms": warm_ms, "achieved_gflops": gflops,
                               "pct_of_peak": pct, "status": "ok"})
                    print(f"  in0_block_w={in0_block_w:4d} subblock={out_subblock_h}x{out_subblock_w}  "
                          f"warm={warm_ms:9.3f}ms  {gflops:10.1f} GFLOPS ({pct:5.1f}% of peak)")
                except Exception as e:
                    write_row({"size": size, "in0_block_w": in0_block_w, "out_subblock_h": out_subblock_h,
                               "out_subblock_w": out_subblock_w, "per_core_M": per_core_M,
                               "per_core_N": per_core_N, "warm_ms": None, "achieved_gflops": None,
                               "pct_of_peak": None, "status": f"skipped: {e}"})
            if tried >= args.max_combos:
                break

        ttnn.deallocate(a_dev)
        ttnn.deallocate(b_dev)
    finally:
        csv_file.close()
        ttnn.close_device(device)

    print(f"\nSaved sweep results to {args.csv}. Compare pct_of_peak across rows: if the best "
          f"manual combo clears ~80%+ while the 'auto-select' baseline row is stuck near the "
          f"originally-observed bad value, the auto-selection heuristic is the culprit for this size.")


if __name__ == "__main__":
    main()