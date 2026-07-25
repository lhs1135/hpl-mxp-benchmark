#!/usr/bin/env python3
"""
run_bench.py — SLURM-based problem-size sweep for hpl-ai.py, one backend at
a time (cpu / gpu / tt), with per-backend sbatch resource requests.

"tt" is this script's name for the Tenstorrent backend; it maps to
hpl-ai.py's own `--device ttnn` flag (see HPL_AI_DEVICE below) — hpl-ai.py
itself still calls it "ttnn", only this sweep script's target name differs.

Submit mode (default):
    Reads a problem-size file (one "N MAX_IT" pair per line), generates a
    self-contained sbatch script per --target backend under a date-stamped
    output folder, and submits each with `sbatch`. Each generated job loops
    over every (N, MAX_IT) x --iterations combination SEQUENTIALLY inside
    the single node allocation (not as separate SLURM jobs), calling:

        python3 hpl-ai.py N MAX_IT --device {ttnn|cpu|gpu} \\
            --metrics-file {outdir}/{target}_{N}_{iter}.txt

    A run that exits non-zero is logged to failed_runs.txt and the sweep
    continues — this runs for up to 24h, a single bad run shouldn't abort
    everything scheduled after it.

    At the end of the sweep, the job calls back into THIS script in
    --summarize mode, so summary.csv appears automatically once the SLURM
    job finishes — no external polling required.

Summarize mode (--summarize DIR, also invoked automatically above):
    Scans DIR for files named "<backend>_<n>_<iter>.txt", parses their
    key=value content (the same format hpl-ai.py's --metrics-file writes),
    and writes DIR/summary.csv, one row per run.

Usage:
    python3 run_bench.py --target cpu --sizes-file sizes.txt
    python3 run_bench.py --target cpu,gpu,tt --sizes-file sizes.txt --iterations 3
    python3 run_bench.py --target tt --sizes-file sizes.txt --dry-run
    python3 run_bench.py --summarize bench_slurm/runs/bench_20260101_120000/cpu

Run with --help for the full option reference.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

VALID_TARGETS = ("cpu", "gpu", "tt")

# This script's target name -> hpl-ai.py's --device value. Only "tt" differs.
HPL_AI_DEVICE = {"cpu": "cpu", "gpu": "gpu", "tt": "ttnn"}

# Per-backend sbatch resource lines, as given for this cluster.
SBATCH_DEVICE_LINES = {
    "cpu": ["-w nvidia-l40s", "-p techfee", "--mem=32G", "--cpus-per-task=16"],
    "gpu": ["-w nvidia-l40s", "-p techfee", "--mem=32G", "--cpus-per-task=16", "-G 1"],
    "tt":  ["-w nvidia-l40s", "-p techfee", "--mem=32G", "--cpus-per-task=16",
            "--gres=r5accel:p150a:1"],
}

# Common to every backend. Note: SLURM directives aren't cumulative — two
# separate `--mail-type=` lines would have the second silently win, not
# combine — so BEGIN/END are merged into one line here.
SBATCH_COMMON_LINES = [
    "--time=0-24:00:00",
    "--mail-type=BEGIN,END",
    "--mail-user=hlim325@gatech.edu",
]

# Per-backend environment setup, run before the benchmark loop inside the
# sbatch job. Override per invocation with --setup-cmd.
DEFAULT_SETUP_CMDS = {
    "cpu": "",
    "gpu": "source ~/set_environment",
    "tt":  "source ~/set_tt_environment",
}

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"
DEFAULT_HPL_AI = Path(__file__).resolve().parent.parent / "hpl-ai.py"

_METRICS_FILENAME_RE = re.compile(r"^(cpu|gpu|tt)_(\d+)_(\d+)\.txt$")


def parse_sizes_file(path: str) -> list[tuple[int, int]]:
    """Parse a 'N MAX_IT' per line file. Blank lines and '#' comments skipped."""
    sizes = []
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{path}:{lineno}: expected 'N MAX_IT', got: {line!r}")
            try:
                n, max_it = int(parts[0]), int(parts[1])
            except ValueError:
                raise ValueError(f"{path}:{lineno}: N and MAX_IT must be integers: {line!r}")
            sizes.append((n, max_it))
    if not sizes:
        raise ValueError(f"{path}: no problem sizes found")
    return sizes


def build_sbatch_script(
    target: str,
    sizes: list[tuple[int, int]],
    iterations: int,
    outdir: Path,
    hpl_ai_script: Path,
    this_script: Path,
    setup_cmd: str,
) -> str:
    lines = ["#!/bin/bash"]
    lines.append(f"#SBATCH --job-name=hplai_{target}")
    for l in SBATCH_DEVICE_LINES[target]:
        lines.append(f"#SBATCH {l}")
    for l in SBATCH_COMMON_LINES:
        lines.append(f"#SBATCH {l}")
    lines.append(f"#SBATCH --output={outdir}/slurm_%j.out")
    lines.append("")
    lines.append("set -uo pipefail  # no -e: one failed run must not abort the whole sweep")
    lines.append("")

    if setup_cmd:
        lines.append("# --- environment setup ---")
        lines.append(setup_cmd)
    else:
        lines.append(
            "# NOTE: no environment setup command for this backend (none needed, "
            "or none given via --setup-cmd)."
        )
    lines.append("")

    lines.append(f'OUTDIR="{outdir}"')
    lines.append('mkdir -p "$OUTDIR"')
    lines.append('FAILED_LOG="$OUTDIR/failed_runs.txt"')
    lines.append(': > "$FAILED_LOG"')
    lines.append("")

    lines.append("SIZES=(")
    for n, max_it in sizes:
        lines.append(f'  "{n} {max_it}"')
    lines.append(")")
    lines.append("")

    lines.append('for pair in "${SIZES[@]}"; do')
    lines.append('  read -r N MAXIT <<< "$pair"')
    lines.append(f"  for i in $(seq 1 {iterations}); do")
    lines.append(f'    echo "=== target={target} N=$N max_it=$MAXIT iter=$i ==="')
    lines.append(
        f'    if ! python3 "{hpl_ai_script}" "$N" "$MAXIT" --device {HPL_AI_DEVICE[target]} \\'
    )
    lines.append(f'         --metrics-file "$OUTDIR/{target}_${{N}}_${{i}}.txt"; then')
    lines.append(f'      echo "{target}_${{N}}_${{i}}" >> "$FAILED_LOG"')
    lines.append('      echo "  [FAILED] N=$N max_it=$MAXIT iter=$i" >&2')
    lines.append("    fi")
    lines.append("  done")
    lines.append("done")
    lines.append("")

    lines.append(f'python3 "{this_script}" --summarize "$OUTDIR"')
    lines.append("")
    return "\n".join(lines)


def parse_metrics_file(path: Path) -> dict[str, str]:
    metrics = {}
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            metrics[k] = v
    return metrics


def summarize(outdir_str: str) -> None:
    outdir = Path(outdir_str)
    if not outdir.is_dir():
        print(f"error: {outdir} is not a directory", file=sys.stderr)
        sys.exit(1)

    rows = []
    all_keys: set[str] = set()
    for path in sorted(outdir.glob("*.txt")):
        m = _METRICS_FILENAME_RE.match(path.name)
        if not m:
            continue  # e.g. failed_runs.txt, or an unrelated file
        backend, n, it = m.group(1), int(m.group(2)), int(m.group(3))
        row = {"backend": backend, "n": n, "iter": it, "source_file": path.name}
        row.update(parse_metrics_file(path))
        rows.append(row)
        all_keys.update(row.keys())

    if not rows:
        print(f"No metrics files found in {outdir} matching '<backend>_<n>_<iter>.txt'.")
        return

    rows.sort(key=lambda r: (r["backend"], r["n"], r["iter"]))
    lead = ["backend", "n", "iter", "source_file"]
    fieldnames = lead + sorted(k for k in all_keys if k not in lead)

    csv_path = outdir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    failed_log = outdir / "failed_runs.txt"
    failed_count = sum(1 for line in open(failed_log) if line.strip()) if failed_log.exists() else 0

    print(f"Summarized {len(rows)} run(s) -> {csv_path}")
    if failed_count:
        print(f"  {failed_count} run(s) failed and have no metrics file — see {failed_log}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_bench.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="comma-separated backend(s) to benchmark: cpu,gpu,tt ('tt' maps "
             "to hpl-ai.py's --device ttnn). One sbatch job is submitted per "
             "backend. Required unless --summarize is given.",
    )
    parser.add_argument(
        "--sizes-file", type=str, default=None,
        help="path to a problem-size file: one 'N MAX_IT' pair per line "
             "(blank lines and '#' comments ignored). Required unless "
             "--summarize is given.",
    )
    parser.add_argument(
        "--iterations", type=int, default=1,
        help="how many times to consecutively repeat each problem size "
             "(default: 1). Metrics files are named "
             "<backend>_<n>_<iter>.txt, iter in 1..N.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"parent directory for this run's date-stamped output folder "
             f"(default: {DEFAULT_RUNS_DIR}). Ignored in --summarize mode.",
    )
    parser.add_argument(
        "--hpl-ai-script", type=str, default=str(DEFAULT_HPL_AI),
        help=f"path to hpl-ai.py (default: {DEFAULT_HPL_AI}).",
    )
    parser.add_argument(
        "--setup-cmd", type=str, default=None,
        help="shell command to run before the benchmark loop inside the "
             "sbatch job, overriding the per-backend default "
             f"({DEFAULT_SETUP_CMDS}). Applies to every --target in this "
             "invocation. Inserted verbatim, once per generated script.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="write the generated sbatch script(s) but don't submit them.",
    )
    parser.add_argument(
        "--summarize", type=str, default=None, metavar="DIR",
        help="skip submission; scan DIR for '<backend>_<n>_<iter>.txt' "
             "metrics files and write DIR/summary.csv. This is also "
             "invoked automatically at the end of each submitted job.",
    )
    args = parser.parse_args()

    if args.summarize is None:
        if not args.target or not args.sizes_file:
            parser.error("--target and --sizes-file are required unless --summarize is given")
        for t in args.target.split(","):
            if t.strip() not in VALID_TARGETS:
                parser.error(f"invalid --target {t.strip()!r}; choose from {VALID_TARGETS}")

    return args


def main() -> None:
    args = _parse_args()

    if args.summarize is not None:
        summarize(args.summarize)
        return

    sizes = parse_sizes_file(args.sizes_file)

    hpl_ai_script = Path(args.hpl_ai_script).resolve()
    if not hpl_ai_script.exists():
        print(f"error: hpl-ai.py not found at {hpl_ai_script} — pass --hpl-ai-script",
              file=sys.stderr)
        sys.exit(1)
    this_script = Path(__file__).resolve()

    targets = [t.strip() for t in args.target.split(",") if t.strip()]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_RUNS_DIR
    batch_dir = base / f"bench_{stamp}"

    print(f"Problem sizes ({len(sizes)}): {sizes}")
    print(f"Iterations per size: {args.iterations}")
    print(f"Output folder: {batch_dir}\n")

    for target in targets:
        outdir = batch_dir / target
        outdir.mkdir(parents=True, exist_ok=True)

        setup_cmd = args.setup_cmd if args.setup_cmd is not None else DEFAULT_SETUP_CMDS[target]
        script_text = build_sbatch_script(
            target, sizes, args.iterations, outdir, hpl_ai_script, this_script,
            setup_cmd,
        )
        sbatch_path = outdir / "submit.sbatch"
        sbatch_path.write_text(script_text)
        print(f"[{target}] wrote {sbatch_path}")

        if args.dry_run:
            continue

        result = subprocess.run(["sbatch", str(sbatch_path)], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[{target}] sbatch submission FAILED:\n{result.stderr}", file=sys.stderr)
        else:
            print(f"[{target}] {result.stdout.strip()}")

    if args.dry_run:
        print(f"\nDry run — no jobs submitted. Review the script(s) under "
              f"{batch_dir}, then rerun without --dry-run.")


if __name__ == "__main__":
    main()
