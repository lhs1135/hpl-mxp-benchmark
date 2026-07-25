# bench_slurm — SLURM problem-size sweep for hpl-ai.py

`run_bench.py` sweeps [hpl-ai.py](../hpl-ai.py) over a list of problem sizes
on a SLURM cluster, one backend (`cpu` / `gpu` / `tt`) at a time, and
gathers the results into a single CSV once the job finishes.

`tt` is this script's name for the Tenstorrent backend; it maps to
hpl-ai.py's own `--device ttnn` flag — hpl-ai.py itself still calls it
`ttnn`, only this sweep script's `--target`/filenames use `tt`.

## What it does

1. You give it a target backend (or several, comma-separated) and a
   problem-size file.
2. For each target, it generates one self-contained `sbatch` script and
   submits it. The script requests resources matching the backend
   (see [Per-backend resources](#per-backend-resources)) and loops over
   every `(N, MAX_IT)` pair from your sizes file — repeated `--iterations`
   times each, consecutively, inside the single node allocation (not as
   separate SLURM jobs).
3. Each run calls `hpl-ai.py N MAX_IT --device {cpu|gpu|ttnn}
   --metrics-file {outdir}/{target}_{N}_{iter}.txt` — reusing hpl-ai.py's
   own machine-readable metrics output (see the root
   [README](../README.md#machine-readable-metrics-file)).
4. A run that exits non-zero is logged to `failed_runs.txt`; the sweep
   keeps going (a bad run partway through a 24h job shouldn't abort
   everything scheduled after it).
5. At the end of the sweep, the job calls back into `run_bench.py
   --summarize` on its own output folder, which parses every
   `<backend>_<n>_<iter>.txt` file and writes `summary.csv` — so the CSV
   is there once the SLURM job completes, with no need to poll `squeue`.

## Usage

```bash
# One backend
python3 run_bench.py --target cpu --sizes-file sizes_example.txt

# Multiple backends in one invocation — submits one job per backend
python3 run_bench.py --target cpu,gpu,tt --sizes-file sizes_example.txt

# Repeat each size 3 times consecutively
python3 run_bench.py --target tt --sizes-file sizes_example.txt --iterations 3

# Review the generated sbatch script(s) without submitting
python3 run_bench.py --target tt --sizes-file sizes_example.txt --dry-run

# Override the per-backend environment setup (see below) for this invocation
python3 run_bench.py --target tt --sizes-file sizes_example.txt \
    --setup-cmd "source /path/to/other_tt_environment"

# Re-summarize an existing output folder by hand (also runs automatically
# at the end of each job)
python3 run_bench.py --summarize runs/bench_20260101_120000/cpu
```

Run `python3 run_bench.py --help` for the full option reference.

### Sizes file format

One `N MAX_IT` pair per line (matches `hpl-ai.py`'s positional args); blank
lines and `#` comments are ignored:

```
# N MAX_IT
50 40
100 40
200 40
```

See [sizes_example.txt](sizes_example.txt).

## Per-backend resources

| Target | sbatch resources | environment setup (`--setup-cmd` default) |
|---|---|---|
| `cpu` | `-w nvidia-l40s -p techfee --mem=32G --cpus-per-task=16` | *(none)* |
| `gpu` | `-w nvidia-l40s -p techfee --mem=32G --cpus-per-task=16 -G 1` | `source ~/set_environment` |
| `tt` | `-w nvidia-l40s -p techfee --mem=32G --cpus-per-task=16 --gres=r5accel:p150a:1` | `source ~/set_tt_environment` |

The environment setup command runs once, near the top of the generated
sbatch script, before the benchmark loop — pass `--setup-cmd` to override
it for all targets in a given invocation.

Every job additionally gets:

```
#SBATCH --time=0-24:00:00
#SBATCH --output={outdir}/slurm_%j.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=hlim325@gatech.edu
```

(SLURM directives aren't cumulative — separate `--mail-type=BEGIN` and
`--mail-type=END` lines would have the second silently win rather than
combine, so these are merged into one `--mail-type=BEGIN,END` line.)

## Output layout

```
runs/bench_<YYYYmmdd_HHMMSS>/
  cpu/
    submit.sbatch
    slurm_<jobid>.out
    cpu_50_1.txt
    cpu_100_1.txt
    ...
    failed_runs.txt      # only present if a run failed
    summary.csv
  gpu/
    ...
  tt/
    ...
```

Each target gets its own subfolder under one date-stamped batch directory
(shared across targets given in the same `--target a,b,c` invocation), so
runs from different sweeps never collide and it's obvious which jobs
belong together.

## summary.csv columns

`backend`, `n`, `iter`, `source_file`, plus every key from hpl-ai.py's
metrics file for that run (`device`, `max_it`, `nb`, `time_*_ms`,
`lu_*_ms`/`lu_*_pct`, `gmres_*`, `gflops`, `eps`, `scaled_residual`,
`scaled_residual_passed`, ...). One row per run — `iter` distinguishes
repeats of the same `(backend, n)` when `--iterations` > 1. Note `backend`
here is `tt`, the value hpl-ai.py itself reports as `device=ttnn` inside
the metrics file (`ttnn`, not `tt`) — the CSV's own `backend` column always
matches the filename/`--target`, not hpl-ai.py's internal device string.

## Requirements

- A SLURM cluster with `sbatch` on `PATH`.
- Whatever `hpl-ai.py` itself needs for the chosen backend (see the root
  [README](../README.md#requirements)) — set up via the per-backend
  defaults above or `--setup-cmd`.
