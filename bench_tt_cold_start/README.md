# Blackhole ttnn Cold-Start / Kernel-Cache Probe

Diagnoses the "first matmul call at a given size is much slower, then fast from the second call on" pattern on Tenstorrent Blackhole (ttnn), and quantifies whether that gap is safe to exclude from a benchmark or not.

## What it does

For each matrix size (square `M=K=N=size`):

1. Snapshots the kernel build/cache directory.
2. Times a **cold** `ttnn.matmul` call (first call on freshly allocated device tensors of that shape).
3. Snapshots the cache directory again and diffs it — new files appearing here means ttnn compiled a new kernel/program config for this shape, not just paid a fixed constant JIT tax.
4. Times a **warm** call (second call, same device tensors, same shape/config — program-cache hit).
5. Writes `cold_ms`, `warm_ms`, `ratio`, `new_cache_files`, `new_cache_bytes` to CSV.

## Why look at the cache directory at all

A flat/constant JIT compile cost predicts the *opposite* of what a "slowdown past a size threshold" looks like — a fixed cost should matter more for small ops, not less. So a threshold-triggered slowdown is more likely explained by ttnn switching to a **different program config** (different kernel set, e.g. interleaved → sharded, 1D → 2D reuse) once the shape crosses some internal heuristic boundary. Watching the cache directory for new files at exactly the sizes where the ratio jumps is a direct way to confirm or rule that out.

## Usage

```bash
# Sweep sizes in one process
python bench_cache_effect.py --sizes 256,512,1024,2048,4096 --csv sweep.csv

# Point at your actual kernel cache location if auto-detection misses it
python bench_cache_effect.py --sizes 256,512,1024,2048,4096 \
    --cache-dirs /path/to/built,/path/to/other --csv sweep.csv
```

Auto-detected cache dir candidates: `$TT_METAL_CACHE`, `$TT_METAL_HOME/built`, `~/.cache/tenstorrent`, `~/.cache/tt-metal-cache`. These vary by tt-metal version/install — pass `--cache-dirs` explicitly if none exist on your system.

### Fresh-process mode (`--single-size`)

An in-process warm/cold pair can't tell apart two different causes:
- cold because ttnn's **in-process** Program Cache is empty this run (would go away on a fresh process if the on-disk kernel cache is intact), vs.
- cold because the **on-disk** kernel binary cache itself doesn't have this shape yet (would still be cold on a fresh process the first time, but fast on the *next* fresh process).

To separate them, run the same size from multiple brand-new processes:

```bash
for i in 1 2 3; do
    python bench_cache_effect.py --single-size 4096 --csv probe.csv
done
```

If run #1 in `probe.csv` is cold and #2/#3 are already warm despite each being a fresh process, the on-disk cache is persisting correctly across restarts. If every fresh process is cold, either the cache isn't persisting (cleared between runs, or pointed at an ephemeral path) or compilation isn't actually the cause.

## The amortized-latency table (instead of a binary include/exclude call)

"Exclude cold start, report steady-state" only makes sense if your real workload calls the same shape many times before the process/cache resets. When the gap is unusually large (15-20x, as observed here) or your deployment doesn't actually reuse shapes that often, that assumption needs checking rather than assuming.

For each size, the script prints and logs an amortized average latency for several reuse counts `N`:

```
avg_latency(N) = (cold_ms + (N - 1) * warm_ms) / N
```

Pick the `N` that matches how often you'd actually call that shape in production (e.g. `N=1` for a cold process per request, `N=1000` for a long-running server on a fixed shape) and read off the honest expected number instead of defaulting to either extreme. These are written to the CSV as `amortized_ms_at_reuse_{N}` for `N in {1, 2, 5, 10, 50, 100, 1000}` (override with `--reuse-counts`).

## Output columns (CSV)

| Column | Meaning |
|---|---|
| `size` | M=K=N for this matmul |
| `cold_ms` | First-call time (includes any compile/cache-miss cost) |
| `warm_ms` | Second-call time on the same tensors |
| `ratio` | `cold_ms / warm_ms` |
| `new_cache_files` / `new_cache_bytes` | New files written to the watched cache dirs between the cold call's before/after snapshots |
| `amortized_ms_at_reuse_N` | Average latency if this shape is called N times before a reset |

## Requirements

- `torch`, `ttnn` (Tenstorrent SDK), `numpy`
