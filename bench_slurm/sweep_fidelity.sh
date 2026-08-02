#!/usr/bin/env bash
# sweep_fidelity.sh — sweep hpl-ai.py's --device ttnn LU factorization over
# every combination of MathFidelity x fp32_dest_acc_en x packer_l1_acc,
# for a list of problem sizes, plus one unconfigured 'default' baseline run
# per size.
#
# Motivation: tt_matmul.py already sweeps these three axes (see its
# make_compute_kernel_config / fidelities list) for an ISOLATED matmul
# microbenchmark. hpl-ai.py now exposes the same axes
# (--fidelity/--fp32-dest-acc/--packer-l1-acc, see
# _resolve_compute_kernel_config) so this sweep can be run against the
# REAL solver workload instead — same reasoning as
# compare_grid_modes.sh for --core-grid-mode.
#
# 4 fidelities (LoFi, HiFi2, HiFi3, HiFi4) x 2 fp32_dest_acc (on/off) x
# 2 packer_l1_acc (on/off) = 16 tuned combinations per size, plus 1
# 'default' (unconfigured) baseline row per size for comparison.
#
# Cache effects (see ../core_grid_experiment/Report.md /
# Tenstorrent_Cache_Theory.md) are keyed per (shape, dtype, core_grid,
# program_config) -- i.e. per COMBINATION here, not per whole-sweep run.
# So the first invocation of a given combination pays a one-time kernel-
# compile cost that has nothing to do with steady-state throughput, and
# ruling that out means repeating each combination itself, not the sweep
# as a whole. REPEATS (env var, default 1) repeats every combination that
# many times and keeps only the LAST repeat's result -- earlier repeats
# overwrite the same metrics file in place and are discarded.
#
# Usage:
#   ./sweep_fidelity.sh                  # default sizes: 1024 2048 4096
#   ./sweep_fidelity.sh 1024 2048
#   MAX_IT=40 OUTDIR=my_results ./sweep_fidelity.sh
#   REPEATS=3 ./sweep_fidelity.sh 1024   # rule out cache cold-start, keep only the 3rd run
#
# Requires: hpl-ai.py runnable with --device ttnn (torch/ttnn installed, a
# Tenstorrent device reachable at device_id=0).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HPL_AI="$REPO_ROOT/hpl-ai.py"

if [ ! -f "$HPL_AI" ]; then
    echo "error: hpl-ai.py not found at $HPL_AI" >&2
    exit 1
fi

SIZES=("$@")
if [ ${#SIZES[@]} -eq 0 ]; then
    SIZES=(1024 2048 4096)
fi

MAX_IT="${MAX_IT:-40}"
REPEATS="${REPEATS:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/results/fidelity_sweep_$STAMP}"
mkdir -p "$OUTDIR"

FIDELITIES=(LoFi HiFi2 HiFi3 HiFi4)
FP32_ACC_OPTS=(on off)
PACKER_L1_OPTS=(on off)

echo "Sizes:         ${SIZES[*]}"
echo "MAX_IT:        $MAX_IT"
echo "Fidelities:    ${FIDELITIES[*]}"
echo "fp32-dest-acc: ${FP32_ACC_OPTS[*]}"
echo "packer-l1-acc: ${PACKER_L1_OPTS[*]}"
echo "Runs per size: $(( ${#FIDELITIES[@]} * ${#FP32_ACC_OPTS[@]} * ${#PACKER_L1_OPTS[@]} + 1 )) (16 tuned + 1 default baseline)"
echo "Repeats per combination: $REPEATS $([ "$REPEATS" -gt 1 ] && echo "(cold-start rule-out; only the last is kept)")"
echo "Output dir:    $OUTDIR"
echo

get_metric() {  # get_metric FILE KEY -> prints value or "n/a"
    local v
    v="$(grep -m1 "^$2=" "$1" 2>/dev/null | cut -d= -f2-)"
    [ -n "$v" ] && echo "$v" || echo "n/a"
}

SUMMARY="$OUTDIR/comparison.csv"
echo "n,fidelity,fp32_dest_acc,packer_l1_acc,time_lu_factorization_ms,lu_strsm_ms,lu_sgemm_ms,gflops" \
    > "$SUMMARY"

run_and_record() {  # run_and_record N FIDELITY FP32 PACKER METRICS_FILE [extra hpl-ai.py args...]
    local n="$1" fid="$2" fp32="$3" packer="$4" metrics_file="$5"
    shift 5
    echo "=== N=$n fidelity=$fid fp32_dest_acc=$fp32 packer_l1_acc=$packer ==="
    # REPEATS>1: each repeat overwrites the same metrics_file in place, so
    # only the LAST one survives -- this combination's (shape, dtype,
    # core_grid, program_config) kernel cache is warmed by the earlier,
    # discarded repeats. See the REPEATS note near the top of this file.
    for rep in $(seq 1 "$REPEATS"); do
        if [ "$REPEATS" -gt 1 ]; then
            echo "  -- repeat $rep/$REPEATS --"
        fi
        python3 "$HPL_AI" "$n" "$MAX_IT" --device ttnn --metrics-file "$metrics_file" "$@"
    done
    echo

    local lu strsm sgemm gf
    lu="$(get_metric "$metrics_file" time_lu_factorization_ms)"
    strsm="$(get_metric "$metrics_file" lu_strsm_ms)"
    sgemm="$(get_metric "$metrics_file" lu_sgemm_ms)"
    gf="$(get_metric "$metrics_file" gflops)"
    echo "$n,$fid,$fp32,$packer,$lu,$strsm,$sgemm,$gf" >> "$SUMMARY"
}

for N in "${SIZES[@]}"; do
    # Baseline: unconfigured, for direct comparison against the tuned combos.
    run_and_record "$N" "default" "n/a" "n/a" "$OUTDIR/ttnn_default_${N}.txt"

    for FID in "${FIDELITIES[@]}"; do
        for FP32 in "${FP32_ACC_OPTS[@]}"; do
            for PACKER in "${PACKER_L1_OPTS[@]}"; do
                METRICS_FILE="$OUTDIR/ttnn_${FID}_${FP32}_${PACKER}_${N}.txt"
                run_and_record "$N" "$FID" "$FP32" "$PACKER" "$METRICS_FILE" \
                    --fidelity "$FID" --fp32-dest-acc "$FP32" --packer-l1-acc "$PACKER"
            done
        done
    done
done

echo
echo "=================================================================================="
printf "%-8s %-10s %-6s %-6s %20s %10s\n" "N" "Fidelity" "fp32" "pkl1" "LU factor (ms)" "GFLOPS"
echo "----------------------------------------------------------------------------------"
tail -n +2 "$SUMMARY" | while IFS=, read -r n fid fp32 packer lu strsm sgemm gf; do
    printf "%-8s %-10s %-6s %-6s %20s %10s\n" "$n" "$fid" "$fp32" "$packer" "$lu" "$gf"
done
echo "=================================================================================="
echo
echo "Full metrics files + comparison.csv written to: $OUTDIR"
