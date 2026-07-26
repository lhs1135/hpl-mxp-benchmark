#!/usr/bin/env python3
"""
compute_efficiency.py — adds "% of theoretical peak" columns to the raw
CSVs produced by tt_matmul.py (TT/ttnn) and gpu_matmul.cu (CUDA/cuBLAS),
for both GFLOPS and PCIe H2D/D2H bandwidth.

Why a separate post-processing script instead of baking this into the two
measurement scripts directly:
  - tt_matmul.py ALREADY computes GFLOPS peak/pct_of_peak per row (its own
    peak_gflops()); this script trusts those columns as-is when present,
    and only ADDS the PCIe pct_of_peak columns that script doesn't compute.
  - gpu_matmul.cu computes neither GFLOPS nor PCIe peak/pct_of_peak at all
    (its own header comment says so explicitly: "does NOT hardcode your
    GPU's peak TFLOPS or PCIe generation"); this script fills both in.
  - Computing peak percentages needs no GPU/TT hardware access at all —
    it's pure CSV math — so keeping it as a standalone post-processing step
    means you don't have to re-run either (potentially long, hardware-bound)
    benchmark just to try different peak-spec assumptions.

Peak-spec inputs (see the normalization notes at the bottom of
tt_matmul.py for how to find these numbers for your exact hardware — this
script deliberately does NOT hardcode vendor numbers, same reasoning as
those two files):

  TT GFLOPS peak   : read from tt_matmul.py's own peak_gflops/pct_of_peak
                     columns when present; recomputed from the row's
                     fidelity + core_grid columns (same formula as
                     tt_matmul.py's peak_gflops()) only as a fallback for
                     older CSVs missing those columns. No flag needed.
  GPU GFLOPS peak  : --gpu-tf32-peak-tflops / --gpu-fp32-peak-tflops
                     (look up your EXACT GPU model's TF32 tensor-core and
                     plain-FP32 CUDA-core TFLOPS from its datasheet — these
                     can differ ~8-10x on the same chip, don't mix them up).
  PCIe peak (both) : --pcie-gen {3,4,5,6} + --pcie-lanes N, using the
                     GB/s-per-lane table below (same numbers tt_matmul.py's
                     comment already documents) — or a direct
                     --pcie-peak-gbps / --tt-pcie-peak-gbps /
                     --gpu-pcie-peak-gbps override if you already have the
                     number (e.g. from `lspci -vv` LnkSta, which shows
                     what's actually negotiated — what matters — not just
                     what the card is capable of). Per-platform overrides
                     matter if the TT card and GPU are in different PCIe
                     generations/slots.

Any peak input you don't supply is left blank in the output (with a note
printed) rather than guessed.

Column naming: TT output keeps tt_matmul.py's own "pct_of_peak" column
name for GFLOPS (untouched, not renamed); GPU output uses the new
"pct_of_peak_gflops" name since that CSV has no prior convention to
respect. Both outputs add "h2d_pct_of_peak" / "d2h_pct_of_peak".

Usage:
    python3 compute_efficiency.py --tt-csv tt_tuned.csv --gpu-csv tuned_matmul_cuda.csv \\
        --pcie-gen 4 --pcie-lanes 16 \\
        --gpu-tf32-peak-tflops 82.6 --gpu-fp32-peak-tflops 20.6

    # TT only, PCIe efficiency only (no GPU peak specs needed)
    python3 compute_efficiency.py --tt-csv tt_tuned.csv --pcie-gen 4 --pcie-lanes 16

    # Different PCIe slots/generations for each device
    python3 compute_efficiency.py --tt-csv tt_tuned.csv --gpu-csv tuned_matmul_cuda.csv \\
        --tt-pcie-peak-gbps 15.75 --gpu-pcie-peak-gbps 63.0 \\
        --gpu-tf32-peak-tflops 82.6 --gpu-fp32-peak-tflops 20.6
"""

from __future__ import annotations

import argparse
import csv
import sys

# Same per-matrix-engine TFLOPS-at-1GHz table as tt_matmul.py's
# PEAK_TFLOPS_PER_ENGINE, duplicated here so this script can recompute a
# TT row's peak_gflops from its fidelity/core_grid columns as a fallback
# for CSVs that predate those columns being written.
PEAK_TFLOPS_PER_ENGINE = {"LoFi": 4.0, "HiFi2": 2.0, "HiFi3": 1.33, "HiFi4": 1.0}

# GB/s per PCIe lane per generation, single direction, real payload
# throughput after line-code overhead — same table documented in
# tt_matmul.py's normalization notes.
PCIE_GBPS_PER_LANE = {3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}


def resolve_pcie_peak(gen: int | None, lanes: int | None, direct: float | None) -> float | None:
    if direct is not None:
        return direct
    if gen is not None and lanes is not None:
        per_lane = PCIE_GBPS_PER_LANE[gen]
        return per_lane * lanes
    return None


def _f(row: dict, key: str) -> float | None:
    v = row.get(key)
    if v is None or v == "" or v == "None":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _tt_peak_gflops_fallback(row: dict) -> float | None:
    per_engine = PEAK_TFLOPS_PER_ENGINE.get(row.get("fidelity", ""))
    if per_engine is None:  # e.g. "default(unconfigured)"
        return None
    grid = row.get("core_grid", "")
    if "x" not in grid:
        return None
    try:
        gx, gy = (int(p) for p in grid.split("x", 1))
    except ValueError:
        return None
    return per_engine * gx * gy * 1000.0


# Same "still suspect" bar tt_matmul.py already uses for its own GFLOPS
# sanity check: real efficiency should land under 100% of peak, not over
# it. >100% almost always means a wrong/mismatched peak-spec input (e.g.
# GPU numbers from one card measured against another's datasheet TFLOPS),
# not a genuinely superhuman result — worth a loud warning either way.
PLAUSIBILITY_WARN_THRESHOLD_PCT = 100.0


def _warn_if_implausible(platform: str, row: dict, metric: str, pct: float) -> None:
    if pct > PLAUSIBILITY_WARN_THRESHOLD_PCT:
        label = row.get("fidelity") or row.get("mode") or "?"
        print(
            f"  [Warning] {platform} size={row.get('size','?')} {label}: {metric}="
            f"{pct:.1f}% of peak exceeds 100% — check the peak-spec inputs "
            f"(likely mismatched hardware or a stale/wrong TFLOPS or GB/s figure).",
            file=sys.stderr,
        )


def _add_pcie_pct_columns(platform: str, rows: list[dict], pcie_peak: float | None) -> None:
    for row in rows:
        for bw_key, pct_key, metric in (
            ("h2d_gbps", "h2d_pct_of_peak", "h2d"), ("d2h_gbps", "d2h_pct_of_peak", "d2h"),
        ):
            bw = _f(row, bw_key)
            if pcie_peak is None or bw is None:
                row[pct_key] = ""
                continue
            pct = bw / pcie_peak * 100
            row[pct_key] = f"{pct:.2f}"
            _warn_if_implausible(platform, row, metric, pct)


def _write_csv(rows: list[dict], fieldnames: list[str], out_path: str) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_tt(path: str, out_path: str, pcie_peak: float | None) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        print(f"warning: {path} has no rows", file=sys.stderr)
        return []

    has_peak_cols = "pct_of_peak" in fieldnames
    if not has_peak_cols:
        for row in rows:
            peak = _tt_peak_gflops_fallback(row)
            achieved = _f(row, "achieved_gflops")
            row["peak_gflops"] = "" if peak is None else f"{peak:.2f}"
            row["pct_of_peak"] = (
                "" if peak is None or achieved is None or peak == 0 else f"{achieved / peak * 100:.2f}"
            )
        for extra in ("peak_gflops", "pct_of_peak"):
            if extra not in fieldnames:
                fieldnames.append(extra)

    for row in rows:
        pct = _f(row, "pct_of_peak")
        if pct is not None:
            _warn_if_implausible("TT", row, "gflops", pct)

    _add_pcie_pct_columns("TT", rows, pcie_peak)
    for extra in ("h2d_pct_of_peak", "d2h_pct_of_peak"):
        if extra not in fieldnames:
            fieldnames.append(extra)

    _write_csv(rows, fieldnames, out_path)
    print(f"[tt] wrote {out_path} ({len(rows)} rows)"
          + (" (recomputed peak_gflops/pct_of_peak — not present in input)" if not has_peak_cols else ""))
    if pcie_peak is None:
        print("  note: no PCIe peak given (--pcie-gen/--pcie-lanes/--pcie-peak-gbps/"
              "--tt-pcie-peak-gbps) — h2d_pct_of_peak/d2h_pct_of_peak left blank.")
    return rows


def process_gpu(
    path: str, out_path: str,
    tf32_peak_tflops: float | None, fp32_peak_tflops: float | None,
    pcie_peak: float | None,
) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        print(f"warning: {path} has no rows", file=sys.stderr)
        return []

    mode_peak_tflops = {"tf32_tensorcore": tf32_peak_tflops, "plain_fp32": fp32_peak_tflops}
    for row in rows:
        peak_tflops = mode_peak_tflops.get(row.get("mode", ""))
        achieved = _f(row, "achieved_gflops")
        peak_gflops = None if peak_tflops is None else peak_tflops * 1000.0
        row["peak_gflops"] = "" if peak_gflops is None else f"{peak_gflops:.2f}"
        if peak_gflops is None or achieved is None or peak_gflops == 0:
            row["pct_of_peak_gflops"] = ""
        else:
            pct = achieved / peak_gflops * 100
            row["pct_of_peak_gflops"] = f"{pct:.2f}"
            _warn_if_implausible("GPU", row, "gflops", pct)
    for extra in ("peak_gflops", "pct_of_peak_gflops"):
        if extra not in fieldnames:
            fieldnames.append(extra)

    _add_pcie_pct_columns("GPU", rows, pcie_peak)
    for extra in ("h2d_pct_of_peak", "d2h_pct_of_peak"):
        if extra not in fieldnames:
            fieldnames.append(extra)

    _write_csv(rows, fieldnames, out_path)
    print(f"[gpu] wrote {out_path} ({len(rows)} rows)")
    if tf32_peak_tflops is None or fp32_peak_tflops is None:
        missing = [m for m, v in (("tf32_tensorcore", tf32_peak_tflops), ("plain_fp32", fp32_peak_tflops)) if v is None]
        print(f"  note: no peak TFLOPS given for mode(s) {missing} — pct_of_peak_gflops left "
              f"blank for those rows (pass --gpu-tf32-peak-tflops/--gpu-fp32-peak-tflops).")
    if pcie_peak is None:
        print("  note: no PCIe peak given (--pcie-gen/--pcie-lanes/--pcie-peak-gbps/"
              "--gpu-pcie-peak-gbps) — h2d_pct_of_peak/d2h_pct_of_peak left blank.")
    return rows


def _print_summary(platform: str, rows: list[dict], label_key: str, pct_gflops_key: str) -> None:
    print(f"\n{platform} summary (% of theoretical peak)")
    print(f"{'size':>8} {label_key:>22} {'gflops%':>9} {'h2d%':>8} {'d2h%':>8}")
    for row in rows:
        def fmt(k):
            v = row.get(k, "")
            return f"{v}%" if v not in ("", None) else "n/a"
        print(f"{row.get('size',''):>8} {row.get(label_key,''):>22} "
              f"{fmt(pct_gflops_key):>9} {fmt('h2d_pct_of_peak'):>8} {fmt('d2h_pct_of_peak'):>8}")


def _default_out(path: str) -> str:
    return path[: -len(".csv")] + "_efficiency.csv" if path.endswith(".csv") else path + "_efficiency.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add % of theoretical peak (GFLOPS + PCIe BW) columns to "
                     "tt_matmul.py / gpu_matmul.cu output CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tt-csv", type=str, default=None, metavar="FILE",
                         help="tt_matmul.py output CSV.")
    parser.add_argument("--gpu-csv", type=str, default=None, metavar="FILE",
                         help="gpu_matmul.cu output CSV.")
    parser.add_argument("--tt-out", type=str, default=None, metavar="FILE",
                         help="output path for the TT CSV (default: <tt-csv minus .csv>_efficiency.csv).")
    parser.add_argument("--gpu-out", type=str, default=None, metavar="FILE",
                         help="output path for the GPU CSV (default: <gpu-csv minus .csv>_efficiency.csv).")

    parser.add_argument("--pcie-gen", type=int, choices=sorted(PCIE_GBPS_PER_LANE), default=None,
                         help="PCIe generation actually negotiated (see `lspci -vv` LnkSta), "
                              "used with --pcie-lanes for both tt and gpu unless overridden below.")
    parser.add_argument("--pcie-lanes", type=int, default=None,
                         help="PCIe lane width actually negotiated, used with --pcie-gen.")
    parser.add_argument("--pcie-peak-gbps", type=float, default=None,
                         help="direct PCIe peak GB/s, applied to both tt and gpu unless a "
                              "per-platform override below is also given.")
    parser.add_argument("--tt-pcie-peak-gbps", type=float, default=None,
                         help="PCIe peak GB/s for the TT device specifically, overriding "
                              "--pcie-gen/--pcie-lanes/--pcie-peak-gbps for the tt CSV only.")
    parser.add_argument("--gpu-pcie-peak-gbps", type=float, default=None,
                         help="PCIe peak GB/s for the GPU specifically, overriding "
                              "--pcie-gen/--pcie-lanes/--pcie-peak-gbps for the gpu CSV only.")

    parser.add_argument("--gpu-tf32-peak-tflops", type=float, default=None,
                         help="GPU's TF32 tensor-core peak TFLOPS (dense, sparsity off), "
                              "from the vendor datasheet for your exact GPU model.")
    parser.add_argument("--gpu-fp32-peak-tflops", type=float, default=None,
                         help="GPU's plain FP32 CUDA-core peak TFLOPS, from the vendor datasheet.")
    args = parser.parse_args()

    if not args.tt_csv and not args.gpu_csv:
        parser.error("give at least one of --tt-csv / --gpu-csv")
    if (args.pcie_gen is None) != (args.pcie_lanes is None):
        parser.error("--pcie-gen and --pcie-lanes must be given together")

    shared_pcie_peak = resolve_pcie_peak(args.pcie_gen, args.pcie_lanes, args.pcie_peak_gbps)

    if args.tt_csv:
        tt_pcie_peak = args.tt_pcie_peak_gbps if args.tt_pcie_peak_gbps is not None else shared_pcie_peak
        rows = process_tt(args.tt_csv, args.tt_out or _default_out(args.tt_csv), tt_pcie_peak)
        if rows:
            _print_summary("TT", rows, "fidelity", "pct_of_peak")

    if args.gpu_csv:
        gpu_pcie_peak = args.gpu_pcie_peak_gbps if args.gpu_pcie_peak_gbps is not None else shared_pcie_peak
        rows = process_gpu(
            args.gpu_csv, args.gpu_out or _default_out(args.gpu_csv),
            args.gpu_tf32_peak_tflops, args.gpu_fp32_peak_tflops, gpu_pcie_peak,
        )
        if rows:
            _print_summary("GPU", rows, "mode", "pct_of_peak_gflops")


if __name__ == "__main__":
    main()
