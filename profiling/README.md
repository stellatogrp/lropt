# CVXRO Profiling Suite

Profiling scripts for the CVXRO core (non-training) pipeline. These scripts identify where time and memory are spent in the canonicalization and solve pipeline.

## Quick Start

```bash
# Run all profiling scripts
bash profiling/run_profiling.sh

# Run a specific script
bash profiling/run_profiling.sh hotspots
bash profiling/run_profiling.sh scaling
bash profiling/run_profiling.sh memory
bash profiling/run_profiling.sh canonicalization

# Or run individually
uv run python profiling/profile_hotspots.py
uv run python profiling/profile_scaling.py
uv run python profiling/profile_memory.py
uv run python profiling/profile_canonicalization.py
```

## Scripts

### `profile_canonicalization.py` — Function-level cProfile

Uses `cProfile` to get a top-down view of where time is spent across the full `solve()` pipeline. Tests 7 problem configurations:

- Small/medium/large Ellipsoidal problems (n=5/50/200)
- Multi-uncertainty (Ellipsoidal + Box)
- Different set types (Box, Budget, Polyhedral)

Outputs sorted cumulative/tottime tables. Saves `.prof` files in `results/` for further analysis with `pstats` or `snakeviz`.

### `profile_memory.py` — Memory Profiling

Uses `tracemalloc` to measure per-stage memory allocations:

- Snapshots before/after: canonicalization, torch expression generation, uncertainty removal, solve
- Top memory allocators by file:line
- Compares across problem sizes (n=5, 50, 200)

### `profile_scaling.py` — Scaling Analysis

Measures how canonicalization time scales with:

- Problem dimension `n` (5, 10, 20, 50, 100, 200)
- Number of constraints `m` (2, 5, 10, 20, 50)
- Number of uncertain parameters (1, 2, 3)

Outputs scaling ratio tables. Generates matplotlib plots if available.

### `profile_hotspots.py` — Targeted Micro-benchmarks

Isolates and benchmarks specific suspected bottlenecks:

1. **`reshape_tensor()`**: Benchmarks the optimized permutation-index approach against the old lil_matrix row-by-row loop (kept as a reference baseline)
2. **`get_problem_data()`**: Measures CVXPY internal canonicalization cost as a fraction of total
3. **`generate_torch_expressions()`**: Quantifies the overhead and potential speedup from deferring this call
4. **Variable creation**: Cost of creating `cp.Variable(shape)` in nested loops
5. **`has_unc_param()`**: Parameter iteration cost on repeated calls
6. **Full solve breakdown**: Time per stage as percentage of total `solve()`

### `run_profiling.sh` — Runner Script

Runs all profiling scripts in sequence (hotspots first as the quickest, then scaling, memory, and cProfile) and saves consolidated output to `results/`.

## Output

Results are saved to `profiling/results/`:
- `.prof` files — loadable with `pstats` or `snakeviz`
- `.log` files — full console output from each run
- `.png` files — scaling plots (if matplotlib available)

## Interpreting Results

- **Cumulative time**: Total time spent in a function including all sub-calls
- **Total time**: Time spent in a function excluding sub-calls
- **Scaling ratios**: How much slower a larger problem is compared to the previous size

Key things to look for:
- What fraction of `solve()` is CVXRO overhead vs CVXPY solver?
- How much time does `generate_torch_expressions()` add (it's not needed for solve)?
- Is `reshape_tensor()` still fast after future changes? (Was a major bottleneck before optimization.)
- Where are the memory hotspots?
