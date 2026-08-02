#!/usr/bin/env python3
"""
run_fidelity_sweep.py — SLURM sbatch generator for sweep_fidelity.sh.

Unlike run_bench.py (which drives hpl-ai.py directly across sizes for the
cpu/gpu/tt backends), this generator's payload is sweep_fidelity.sh
itself, which already sweeps all 16 fidelity/fp32-dest-acc/packer-l1-acc
combinations (+ a default baseline) across a list of sizes and writes its
own comparison.csv. Always --device ttnn — fidelity/core-grid/packer_l1
are TT-only concepts, there's no cpu/gpu variant of this sweep, so unlike
run_bench.py there's no --target to choose.

Cache-effect rule-out: sweep_fidelity.sh's kernel/program-config cache is
keyed per (shape, dtype, core_grid, program_config) — i.e. per fidelity
combination here, not per whole-sweep run (see
../core_grid_experiment/Report.md / Tenstorrent_Cache_Theory.md). So each
combination's FIRST invocation pays a one-time kernel-compile cost that
has nothing to do with steady-state throughput. --repeats N (default 3,
passed through as sweep_fidelity.sh's REPEATS env var) repeats EACH
combination N times and keeps only the last repeat's result — automating
what was previously a manual "run it 3 times, throw away everything but
the last" step.

Usage:
    python3 run_fidelity_sweep.py --sizes 1024,2048,4096
    python3 run_fidelity_sweep.py --sizes 1024,2048 --max-it 40 --repeats 5
    python3 run_fidelity_sweep.py --sizes 1024 --dry-run
    python3 run_fidelity_sweep.py --sizes 1024 --clean-tt ~/.cache/tenstorrent
    python3 run_fidelity_sweep.py --sizes 1024 --mail-user you@gatech.edu

Run with --help for the full option reference.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Same sbatch resource line this cluster uses for hpl-ai.py's "tt" target
# in run_bench.py — this generator is always ttnn, so there's no per-
# backend table to pick from.
SBATCH_TT_LINES = [
    "-w nvidia-l40s", "-p techfee", "--mem=32G", "--cpus-per-task=16",
    "--gres=r5accel:p150a:1",
]
SBATCH_COMMON_LINES = ["--time=0-24:00:00"]
DEFAULT_SETUP_CMD = "source ~/set_tt_environment"

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"
DEFAULT_SWEEP_SCRIPT = Path(__file__).resolve().parent / "sweep_fidelity.sh"


def _is_unsafe_clean_dir(path: str) -> bool:
    """Same guard as run_bench.py's --clean-tt — see there for rationale."""
    resolved = os.path.abspath(os.path.expanduser(path.strip())) if path.strip() else ""
    home = os.path.abspath(os.path.expanduser("~"))
    return resolved in ("", "/", home)


def build_sbatch_script(
    sizes: list[int],
    max_it: int,
    repeats: int,
    outdir: Path,
    sweep_script: Path,
    setup_cmd: str,
    mail_user: str | None,
    clean_tt_dir: str | None,
) -> str:
    lines = ["#!/bin/bash"]
    lines.append("#SBATCH --job-name=hplai_fidelity_sweep")
    for l in SBATCH_TT_LINES:
        lines.append(f"#SBATCH {l}")
    for l in SBATCH_COMMON_LINES:
        lines.append(f"#SBATCH {l}")
    lines.append(f"#SBATCH --output={outdir}/slurm_%j.out")
    if mail_user:
        # Merged into one --mail-type line: SLURM directives aren't
        # cumulative, so separate BEGIN/END lines would have the second
        # silently win instead of combining.
        lines.append("#SBATCH --mail-type=BEGIN,END")
        lines.append(f"#SBATCH --mail-user={mail_user}")
    lines.append("")
    lines.append("set -uo pipefail  # no -e: one failed combination must not abort the whole sweep")
    lines.append("")

    if setup_cmd:
        lines.append("# --- environment setup ---")
        lines.append(setup_cmd)
    else:
        lines.append("# NOTE: no environment setup command given (--setup-cmd).")
    lines.append("")

    if clean_tt_dir:
        lines.append("# --- clean tt cache before this sweep (--clean-tt) ---")
        lines.append(f'echo "Cleaning tt cache dir: {clean_tt_dir}"')
        lines.append(f'rm -rf -- "{clean_tt_dir}"')
        lines.append("")

    lines.append(f'mkdir -p "{outdir}"')
    sizes_str = " ".join(str(n) for n in sizes)
    lines.append(f'MAX_IT={max_it} REPEATS={repeats} OUTDIR="{outdir}" \\')
    lines.append(f'    bash "{sweep_script}" {sizes_str}')
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_fidelity_sweep.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sizes", type=str, required=True,
        help="comma-separated problem sizes, e.g. 1024,2048,4096.",
    )
    parser.add_argument(
        "--max-it", type=int, default=40,
        help="GMRES restart length, shared across all sizes (default: 40).",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="repeat each fidelity combination this many times, keeping "
             "only the last result, to rule out kernel-cache cold-start "
             "effects (default: 3 — see module docstring). 1 disables "
             "repetition.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"parent directory for this run's date-stamped output folder "
             f"(default: {DEFAULT_RUNS_DIR}).",
    )
    parser.add_argument(
        "--sweep-script", type=str, default=str(DEFAULT_SWEEP_SCRIPT),
        help=f"path to sweep_fidelity.sh (default: {DEFAULT_SWEEP_SCRIPT}).",
    )
    parser.add_argument(
        "--setup-cmd", type=str, default=DEFAULT_SETUP_CMD,
        help=f"shell command to run before the sweep starts (default: "
             f"{DEFAULT_SETUP_CMD!r}, matching run_bench.py's tt default).",
    )
    parser.add_argument(
        "--mail-user", type=str, default=None, metavar="EMAIL",
        help="email address for SLURM job-start/job-end notifications. "
             "If omitted, no mail directives are added and SLURM sends no mail.",
    )
    parser.add_argument(
        "--clean-tt", type=str, default=None, metavar="DIR",
        help="if given, `rm -rf DIR` before the sweep starts (e.g. a ttnn "
             "kernel/build cache directory, to force every combination cold).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="write the generated sbatch script but don't submit it.",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.clean_tt is not None and _is_unsafe_clean_dir(args.clean_tt):
        parser.error(
            f"--clean-tt {args.clean_tt!r} looks unsafe (empty, '/', or "
            "your home directory) — refusing to generate `rm -rf` for it"
        )
    return args


def main() -> None:
    args = _parse_args()

    try:
        sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    except ValueError:
        print(f"error: --sizes must be comma-separated integers, got {args.sizes!r}", file=sys.stderr)
        sys.exit(1)
    if not sizes:
        print("error: --sizes gave no problem sizes", file=sys.stderr)
        sys.exit(1)

    sweep_script = Path(args.sweep_script).resolve()
    if not sweep_script.exists():
        print(f"error: sweep_fidelity.sh not found at {sweep_script} — pass --sweep-script",
              file=sys.stderr)
        sys.exit(1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_RUNS_DIR
    outdir = base / f"fidelity_sweep_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Sizes: {sizes}")
    print(f"MAX_IT: {args.max_it}")
    print(f"Repeats per combination: {args.repeats}")
    print(f"Output folder: {outdir}\n")

    script_text = build_sbatch_script(
        sizes, args.max_it, args.repeats, outdir, sweep_script,
        args.setup_cmd, args.mail_user, args.clean_tt,
    )
    sbatch_path = outdir / "submit.sbatch"
    sbatch_path.write_text(script_text)
    print(f"wrote {sbatch_path}")

    if args.dry_run:
        print(f"\nDry run — no job submitted. Review {sbatch_path}, then rerun without --dry-run.")
        return

    result = subprocess.run(["sbatch", str(sbatch_path)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"sbatch submission FAILED:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
