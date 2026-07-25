"""
bench_cache_effect.py — Diagnoses the "first run is ~Nx slower, then fast"
pattern on Tenstorrent Blackhole (ttnn) matmul, and gives you the numbers
needed to decide whether excluding cold-start from your benchmark is
actually justified for your workload (see the bottom of this docstring).

What it measures, per matrix size (square MxKxN = size^3):
  1. A snapshot of the kernel build/cache directory before the first call.
  2. cold_ms  : wall-clock time of the FIRST ttnn.matmul call on this exact
                shape/dtype/device tensors (includes any JIT kernel compile
                + program-cache-miss overhead).
  3. A second snapshot of the cache directory right after — diffed against
     (1) to report how many new files / bytes got written. If new files
     appear only when you cross your size threshold, that's a direct,
     file-level confirmation that ttnn switched to a different internal
     program config (different kernel set) at that size, rather than the
     slowdown being generic/constant JIT overhead.
  4. warm_ms : wall-clock time of a SECOND ttnn.matmul call on the SAME
                pre-allocated device tensors (program cache hit, no new
                compile).

Two run modes:

  --sizes "512,1024,2048,4096"
      Normal in-process sweep. Good for finding *where* the threshold is
      and whether it correlates with new cache files.

  --single-size 4096
      Runs cold+warm for exactly ONE size and exits. Meant to be invoked
      from a shell loop, once per size, as a **brand-new process each
      time** (see the loop example below). This is the only way to tell
      apart:
        - "cold because ttnn's in-process Program Cache is empty this run"
          (would disappear across a fresh process if the on-disk kernel
          binary cache is intact and reused)
        - "cold because the on-disk kernel binary cache itself doesn't
          have this shape/config yet" (would still be slow on a fresh
          process the first time, but fast on the NEXT fresh process for
          the same shape, since the disk cache now has it)

      Shell loop example (run the SAME size across 3 fresh processes):
          for i in 1 2 3; do
              python bench_cache_effect.py --single-size 4096 --csv probe.csv
          done
      Then look at probe.csv: if run #1 is cold and #2/#3 are already warm
      even though each was a fresh process, the on-disk cache is doing its
      job and persists across restarts. If EVERY fresh process is cold,
      either the cache directory isn't persisting (e.g. it's being cleared
      between runs, or TT_METAL_CACHE points somewhere ephemeral) or the
      slowdown isn't actually about kernel compilation.

Usage:
    python bench_cache_effect.py --sizes 512,1024,2048,4096,8192 --csv sweep.csv
    python bench_cache_effect.py --single-size 4096 --csv probe.csv
    python bench_cache_effect.py --sizes 512,4096 --cache-dirs /path/to/built,/path/to/other

On "should I just throw away the cold-start number"
-----------------------------------------------------
The standard "exclude cold start, report steady-state" convention assumes
your production workload calls the SAME shape/config many times, so the
one-time compile cost gets amortized away to something negligible. That
assumption breaks down, or at least deserves scrutiny, when:
  - the gap is unusually large (this script exists because a 15-20x gap
    is well outside typical JIT overhead, which is usually a small
    constant relative to steady-state time for non-trivial ops),
  - your real deployment doesn't actually call the same shape many times
    in a row (dynamic shapes, short-lived processes, autoscaled workers
    that restart often), or
  - the on-disk kernel cache doesn't actually survive between the
    processes you deploy (see --single-size above) — in which case
    "warm" is a benchmark fiction that never happens in production.

Instead of a binary include/exclude decision, this script prints an
amortized-latency table per size: assuming a shape is reused N times
before the process/cache resets, average_latency(N) = (cold + (N-1)*warm) / N.
Plug in the N that actually matches how your deployment calls this shape
(N=1 if every call is a cold process, N=1000 if you serve a fixed shape
for a long-running session) and read off the honest expected number
instead of guessing.
"""

import argparse
import csv
import os
import sys
import time


DEFAULT_REUSE_COUNTS = (1, 2, 5, 10, 50, 100, 1000)


def guess_cache_dirs():
    """Best-effort list of candidate tt-metal kernel build/cache directories.
    Exact default location varies by tt-metal version/install, so this is a
    starting point -- pass --cache-dirs explicitly if none of these exist on
    your system (check `TT_METAL_CACHE` / `TT_METAL_HOME` env vars you have
    set, or watch `lsof`/`strace` output during a cold run to find where new
    files actually land)."""
    candidates = []
    if os.environ.get("TT_METAL_CACHE"):
        candidates.append(os.environ["TT_METAL_CACHE"])
    if os.environ.get("TT_METAL_HOME"):
        candidates.append(os.path.join(os.environ["TT_METAL_HOME"], "built"))
    candidates.append(os.path.expanduser("~/.cache/tenstorrent"))
    candidates.append(os.path.expanduser("~/.cache/tt-metal-cache"))
    return [c for c in dict.fromkeys(candidates) if os.path.isdir(c)]


def snapshot_dir(path):
    """Returns {relative_path: size_bytes} for every file under path.
    Cheap-ish (no hashing, no mtime comparisons) so it's safe to call
    twice per benchmarked size without meaningfully perturbing timing."""
    out = {}
    for root, _dirs, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            try:
                out[os.path.relpath(full, path)] = os.path.getsize(full)
            except OSError:
                pass  # file could vanish mid-walk (temp build artifacts); ignore
    return out


def diff_snapshots(before, after):
    new_files = [p for p in after if p not in before]
    new_bytes = sum(after[p] for p in new_files)
    return len(new_files), new_bytes, new_files


def time_matmul(ttnn, torch, device, a_dev, b_dev, dtype, compute_kernel_config):
    t0 = time.perf_counter()
    out_dev = ttnn.matmul(a_dev, b_dev, dtype=dtype, compute_kernel_config=compute_kernel_config)
    _ = ttnn.to_torch(out_dev)  # forces device->host readback, i.e. a real sync point
    t1 = time.perf_counter()
    ttnn.deallocate(out_dev)
    return (t1 - t0) * 1000.0


def amortized_table(cold_ms, warm_ms, reuse_counts):
    return {n: (cold_ms + (n - 1) * warm_ms) / n for n in reuse_counts}


def bench_one_size(ttnn, torch, device, size, dtype, cache_dirs, reuse_counts):
    import numpy as np

    m = k = n = size
    rng = np.random.default_rng(0)
    a_np = rng.uniform(-1, 1, size=(m, k)).astype(np.float32)
    b_np = rng.uniform(-1, 1, size=(k, n)).astype(np.float32)
    a_t = torch.tensor(a_np, dtype=torch.float32)
    b_t = torch.tensor(b_np, dtype=torch.float32)

    a_dev = ttnn.from_torch(a_t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b_dev = ttnn.from_torch(b_t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    cfg_cls = getattr(ttnn, "BlackholeComputeKernelConfig", None) or ttnn.WormholeComputeKernelConfig
    compute_kernel_config = cfg_cls(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    before = {d: snapshot_dir(d) for d in cache_dirs}
    cold_ms = time_matmul(ttnn, torch, device, a_dev, b_dev, dtype, compute_kernel_config)
    after = {d: snapshot_dir(d) for d in cache_dirs}
    warm_ms = time_matmul(ttnn, torch, device, a_dev, b_dev, dtype, compute_kernel_config)

    ttnn.deallocate(a_dev)
    ttnn.deallocate(b_dev)

    new_files_total, new_bytes_total, new_files_by_dir = 0, 0, {}
    for d in cache_dirs:
        nf, nb, files = diff_snapshots(before[d], after[d])
        new_files_total += nf
        new_bytes_total += nb
        new_files_by_dir[d] = files

    row = {
        "size": size,
        "cold_ms": cold_ms,
        "warm_ms": warm_ms,
        "ratio": cold_ms / warm_ms if warm_ms > 0 else float("nan"),
        "new_cache_files": new_files_total,
        "new_cache_bytes": new_bytes_total,
    }
    for reuse_n, amortized in amortized_table(cold_ms, warm_ms, reuse_counts).items():
        row[f"amortized_ms_at_reuse_{reuse_n}"] = amortized

    return row, new_files_by_dir


def main():
    parser = argparse.ArgumentParser(description="Probe Blackhole/ttnn cold-start vs kernel-cache effects")
    parser.add_argument("--sizes", type=str, default="256,512,1024,2048,4096",
                         help="Comma-separated square matmul sizes (M=K=N)")
    parser.add_argument("--single-size", type=int, default=None,
                         help="Run exactly one size and exit (for the fresh-process shell-loop workflow)")
    parser.add_argument("--cache-dirs", type=str, default=None,
                         help="Comma-separated paths to snapshot for new kernel binaries. "
                              "Defaults to auto-detected candidates (see guess_cache_dirs()).")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--csv", type=str, default="bench_cache_effect.csv")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--reuse-counts", type=str, default=",".join(str(n) for n in DEFAULT_REUSE_COUNTS),
                         help="Comma-separated N values for the amortized-latency table")
    args = parser.parse_args()

    try:
        import torch
        import ttnn
    except ImportError as e:
        print(f"[Error] Failed to import torch/ttnn: {e}", file=sys.stderr)
        sys.exit(1)

    cache_dirs = [d.strip() for d in args.cache_dirs.split(",")] if args.cache_dirs else guess_cache_dirs()
    if not cache_dirs:
        print("[Warning] No cache directories found/specified — new-file diffing will report 0 for everything. "
              "Pass --cache-dirs explicitly once you know where your build cache lives.", file=sys.stderr)
    else:
        print(f"Watching cache dirs: {cache_dirs}")

    dtype = ttnn.float32 if args.dtype == "float32" else ttnn.bfloat16
    reuse_counts = tuple(int(x) for x in args.reuse_counts.split(","))
    sizes = [args.single_size] if args.single_size is not None else [int(x) for x in args.sizes.split(",")]

    csv_exists = os.path.exists(args.csv)
    fieldnames = ["size", "cold_ms", "warm_ms", "ratio", "new_cache_files", "new_cache_bytes"] + \
                 [f"amortized_ms_at_reuse_{n}" for n in reuse_counts]

    device = ttnn.open_device(device_id=args.device_id)
    try:
        with open(args.csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not csv_exists:
                writer.writeheader()

            for size in sizes:
                row, new_files_by_dir = bench_one_size(ttnn, torch, device, size, dtype, cache_dirs, reuse_counts)
                writer.writerow(row)
                f.flush()

                print(f"\nsize={size}  cold={row['cold_ms']:.1f}ms  warm={row['warm_ms']:.1f}ms  "
                      f"ratio={row['ratio']:.2f}x  new_cache_files={row['new_cache_files']} "
                      f"({row['new_cache_bytes']} bytes)")
                for d, files in new_files_by_dir.items():
                    if files:
                        preview = files[:5]
                        more = f" (+{len(files) - 5} more)" if len(files) > 5 else ""
                        print(f"    new in {d}: {preview}{more}")

                print("    amortized avg latency by reuse count N (cold paid once, warm for the rest):")
                for n in reuse_counts:
                    print(f"      N={n:>5}: {row[f'amortized_ms_at_reuse_{n}']:.2f} ms")
    finally:
        ttnn.close_device(device)

    print(f"\nAppended results to {args.csv}")


if __name__ == "__main__":
    main()