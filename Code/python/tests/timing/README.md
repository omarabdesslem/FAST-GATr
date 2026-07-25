# Timing Benchmarks

This folder contains timing scripts used to compare the original Python EzGATr
implementation against ASL C++ implementations.

Scripts are split by timing unit:

```text
timing/
├── seconds/
│   ├── baseline_python_code.py
│   ├── baseline_c_code.py
│   ├── optimized_c_code.py
│   ├── optimized_for_cache_c_code.py
│   └── simd_c_code.py
└── cycles/
    ├── baseline_python_cycles.py
    ├── baseline_c_cycles.py
    ├── baseline_per_kernel_cycles.py
    └── _rdtsc_helper.dll
```

The main seconds benchmarks are:

```bash
python3 -m tests.timing.seconds.baseline_python_code
python3 -m tests.timing.seconds.baseline_c_code
python3 -m tests.timing.seconds.optimized_c_code
python3 -m tests.timing.seconds.optimized_for_cache_c_code
python3 -m tests.timing.seconds.simd_c_code
```

Both scripts run the same set of input sizes and write timing results to CSV files under:

```text
../Results/timing_results/CSVs/<YYYY-MM-DD>/
```

---

## 0. Short version: Script

This should always work.

From the `Code/python/` directory:

```bash
python3 -m tests.timing.seconds.baseline_python_code
python3 -m tests.timing.seconds.baseline_c_code
python3 -m tests.timing.seconds.optimized_c_code
python3 -m tests.timing.seconds.optimized_for_cache_c_code
python3 -m tests.timing.seconds.simd_c_code
```

The C++ timing scripts rebuild their own implementation before running:

- `baseline_c_code.py` builds `cpp/baseline`
- `optimized_c_code.py` builds `cpp/optimized`
- `optimized_for_cache_c_code.py` builds `cpp/optimized_for_cache`
- `simd_c_code.py` builds `cpp/simd`

After this, the results directory should contain CSVs for Python, C++, and optimized C++:

```text
../Results/timing_results/CSVs/<YYYY-MM-DD>/python_timing_results_<run-id>.csv
../Results/timing_results/CSVs/<YYYY-MM-DD>/c_timing_results_<run-id>.csv
../Results/timing_results/CSVs/<YYYY-MM-DD>/optimized_c_timing_results_<run-id>.csv
../Results/timing_results/CSVs/<YYYY-MM-DD>/optimized_for_cache_c_timing_results_<run-id>.csv
../Results/timing_results/CSVs/<YYYY-MM-DD>/simd_c_timing_results_<run-id>.csv
```

---


## 1. Step by step: Compilation

The C++ timing scripts compile their target implementation automatically. To
manually build a specific implementation, set `CPP_IMPL`.

From the `Code/python/` directory, run:

```bash
find . -name "*.so" -delete
rm -rf build
CPP_IMPL=baseline python3 setup.py build_ext --inplace
CPP_IMPL=optimized python3 setup.py build_ext --inplace
CPP_IMPL=optimized_for_cache python3 setup.py build_ext --inplace
CPP_IMPL=simd python3 setup.py build_ext --inplace
```

This compiles the C++ extensions such as:

- `linear`
- `dual`
- `attention`
- `norm`

The generated `.so` files are copied into the project directory so Python can import them.

---

## 2. When to Run `make_basis.py`

Usually, you do **not** need to rerun `make_basis.py` before every benchmark.

Run it only if:

- basis-generated files are missing,
- you changed the basis generation code,
- you deleted generated basis files,
- or the test/benchmark explicitly fails because basis files are missing.

If needed, run:

```bash
python3 -m ezgatr_extensions.make_basis
```

Then rebuild the C++ extensions:

```bash
find . -name "*.so" -delete
rm -rf build
python3 setup.py build_ext --inplace
```

---

## 3. Running the Baseline function

The Python baseline uses the original EzGATr implementation.

Run:

```bash
python3 -m tests.timing.seconds.baseline_python_code
```

or 

```bash
python3 -m tests.timing.seconds.baseline_c_code
```

For the optimized C++ timing script, run:

```bash
python3 -m tests.timing.seconds.optimized_c_code
```

For the optimized-for-cache C++ timing script, run:

```bash
python3 -m tests.timing.seconds.optimized_for_cache_c_code
```

For the SIMD C++ timing script, run:

```bash
python3 -m tests.timing.seconds.simd_c_code
```

The results are written to:

```text
../Results/timing_results/CSVs/<YYYY-MM-DD>/python_timing_results_<run-id>.csv
../Results/timing_results/CSVs/<YYYY-MM-DD>/c_timing_results_<run-id>.csv
../Results/timing_results/CSVs/<YYYY-MM-DD>/optimized_c_timing_results_<run-id>.csv
../Results/timing_results/CSVs/<YYYY-MM-DD>/optimized_for_cache_c_timing_results_<run-id>.csv
../Results/timing_results/CSVs/<YYYY-MM-DD>/simd_c_timing_results_<run-id>.csv
```

## 4. Cycle Benchmarks

Cycle-counting scripts live in `tests/timing/cycles/`.

Run full-forward cycle benchmarks from `Code/python/`:

```bash
python3 -m tests.timing.cycles.baseline_python_cycles
python3 -m tests.timing.cycles.baseline_c_cycles
```

Run per-kernel cycle counters after rebuilding with `CYCLE_COUNT=1`:
The script also sets this automatically, but the manual form is:

```bash
CPP_IMPL=baseline CYCLE_COUNT=1 python3 setup.py build_ext --inplace
python3 -m tests.timing.cycles.baseline_per_kernel_cycles
```



## 5. Changing Parameters

The benchmark parameters are defined as global variables near the top of each timing file.

Common variables to edit:

```python
RESULTS_DIR = TEAM80_DIR / "Results" / "timing_results" / "CSVs" / DATE
OUTPUT_FILE = RESULTS_DIR / f"python_timing_results_{DATE_TIME}.csv"
WARMUP = 10
RUNS = 30
SEED = 42
TEST_CASES = [...]
```

You can also add entries to `TEST_CASES`:

```python
TEST_CASES = [
    (1, 1, 1),
    (1, 4, 2),
    (1, 16, 2),
    (8, 256, 2),
]
```



## 6. Compiler Flags

For performance measurements, make sure the C++ extensions are compiled with optimization enabled.

Recommended flags in `setup.py`:

```python
extra_compile_args=[
    "-O3",
    "-std=c++17",
    "-DNDEBUG",
    "-ffast-math",
]
```

On Apple Silicon, you can also try:

```python
extra_compile_args=[
    "-O3",
    "-std=c++17",
    "-DNDEBUG",
    "-ffast-math",
    "-mcpu=native",
]
```

The default build now uses `-mcpu=native` on macOS ARM, which lets an M3
MacBook Air compile for its local CPU. To force an explicit Apple CPU target,
set `ASL_APPLE_MCPU` before rebuilding, for example:

```bash
ASL_APPLE_MCPU=apple-m3 CPP_IMPL=simd python3 setup.py build_ext --inplace --force
```

After changing compiler flags, rebuild:

```bash
find . -name "*.so" -delete
rm -rf build
python3 setup.py build_ext --inplace
```

---

## 7. Plotting and FLOP Helpers

Plotting and FLOP-counting helpers live outside the benchmark package:

```text
Code/python/tools/plots/
Code/python/tools/cpp_flop_counter.py
```

Run them from the repository root:

```bash
python Code/python/tools/plots/cpp_flop_counter_sweep.py
python Code/python/tools/plots/flops_per_cycle.py
python Code/python/tools/plots/plot_flops_per_cycle.py
```
