#!/usr/bin/env bash
# compare_grid_modes.sh — A/B hpl-ai.py's --device ttnn LU factorization
# under --core-grid-mode default vs max, for a list of problem sizes.
#
# Motivation: Report.md / tt_matmul.py found that an isolated, explicitly
# tuned ttnn.matmul (full core grid, fidelity set) closes most of the gap
# against GPU that the LU solver benchmark showed at 49-310x — strong
# evidence the solver's own ttnn.matmul calls (previously no core_grid at
# all) were the actual bottleneck, not Blackhole's raw throughput. hpl-ai.py
# now supports --core-grid-mode {default,max,custom} (see sgetrf_nopiv_ttnn /
# _resolve_core_grid in ../hpl-ai.py) so that hypothesis can be checked
# against the REAL solver workload, not just an isolated matmul microbench.
#
# This script runs the same problem sizes under 'default' and 'max' and
# puts LU-factorization timing side by side, so you can see directly
# whether pinning the full grid actually helps the real solver, and by how
# much.
#
# Usage:
#   ./compare_grid_modes.sh                    # default sizes: 1024 2048 4096
#   ./compare_grid_modes.sh 1024 2048 4096 10240
#   MAX_IT=40 OUTDIR=my_results ./compare_grid_modes.sh
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
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/results/grid_compare_$STAMP}"
mkdir -p "$OUTDIR"

echo "Sizes:      ${SIZES[*]}"
echo "MAX_IT:     $MAX_IT"
echo "Output dir: $OUTDIR"
echo

get_metric() {  # get_metric FILE KEY -> prints value or "n/a"
    local v
    v="$(grep -m1 "^$2=" "$1" 2>/dev/null | cut -d= -f2-)"
    [ -n "$v" ] && echo "$v" || echo "n/a"
}

SUMMARY="$OUTDIR/comparison.csv"
echo "n,lu_factorization_ms_default,lu_factorization_ms_max,speedup,strsm_ms_default,strsm_ms_max,sgemm_ms_default,sgemm_ms_max,gflops_default,gflops_max" \
    > "$SUMMARY"

for N in "${SIZES[@]}"; do
    for MODE in default max; do
        METRICS_FILE="$OUTDIR/ttnn_${MODE}_${N}.txt"
        echo "=== N=$N core-grid-mode=$MODE ==="
        python3 "$HPL_AI" "$N" "$MAX_IT" --device ttnn --core-grid-mode "$MODE" \
            --metrics-file "$METRICS_FILE"
        echo
    done

    DF="$OUTDIR/ttnn_default_${N}.txt"
    MF="$OUTDIR/ttnn_max_${N}.txt"

    LU_D="$(get_metric "$DF" time_lu_factorization_ms)"
    LU_M="$(get_metric "$MF" time_lu_factorization_ms)"
    STRSM_D="$(get_metric "$DF" lu_strsm_ms)"
    STRSM_M="$(get_metric "$MF" lu_strsm_ms)"
    SGEMM_D="$(get_metric "$DF" lu_sgemm_ms)"
    SGEMM_M="$(get_metric "$MF" lu_sgemm_ms)"
    GF_D="$(get_metric "$DF" gflops)"
    GF_M="$(get_metric "$MF" gflops)"

    SPEEDUP="$(python3 -c "
try:
    d, m = float('$LU_D'), float('$LU_M')
    print(f'{d / m:.2f}x' if m > 0 else 'n/a')
except ValueError:
    print('n/a')
")"

    echo "$N,$LU_D,$LU_M,$SPEEDUP,$STRSM_D,$STRSM_M,$SGEMM_D,$SGEMM_M,$GF_D,$GF_M" >> "$SUMMARY"
done

echo
echo "=================================================================="
printf "%-8s %20s %20s %10s\n" "N" "LU default (ms)" "LU max (ms)" "speedup"
echo "------------------------------------------------------------------"
tail -n +2 "$SUMMARY" | while IFS=, read -r n lu_d lu_m speedup _; do
    printf "%-8s %20s %20s %10s\n" "$n" "$lu_d" "$lu_m" "$speedup"
done
echo "=================================================================="
echo
echo "Full metrics files + comparison.csv written to: $OUTDIR"
