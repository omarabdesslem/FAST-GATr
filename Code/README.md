# Code

Project code split by language.

## Structure

- `cpp/` — C++ source code, optimized/baseline kernels, and generated C++ kernel snippets.
- `python/` — Python packages, tests, examples, tools, basis data, and the C++ extension build script.

## Setup

From the Python directory (`Code/python/`):

```bash
pip install -e .
```

For a clean rebuild:

```bash
rm -rf build/ dist/ *.egg-info
pip install -e .
```

## Run example

```bash
python example/main.py
```

## Run tests

```bash
pytest tests/
```

Run one test file:

```bash
pytest tests/validation/baseline/baseline_test_end_to_end.py
```

## Timing Results

Timing scripts write CSV files to the repository-level results directory:

```text
../../Results/timing_results/CSVs/<YYYY-MM-DD>/
```

## Helper Tools

Run helper tools from `Code/python/` unless a script says otherwise:

```bash
python tools/build_fast.py build_ext --inplace --force
python tools/profiling/profile_individual_kernel.py
```
