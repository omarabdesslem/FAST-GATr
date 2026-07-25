# Tests

This directory contains tests, including correctness validation tests and timing benchmarks.

## Structure

```text
tests/
├── validation/
├── timing/
└── README.md
```

## Validation

Validation tests compare the original Python EzGATr implementation with the ASL C++ implementation.

Before running validation, compile the C++ extensions from the `Code/python/` directory:

```bash
python3 setup.py build_ext
```

If basis files are missing, generate them first:

```bash
python3 -m ezgatr_extensions.make_basis
python3 setup.py build_ext
```

Run validation from the `Code/python/` directory:

```bash
python3 -m tests.validation.baseline.baseline_test_end_to_end
python3 -m tests.validation.optimized.optimized_test_end_to_end
```

## Timing

Timing scripts measure runtime for the Python and C++ implementations.

Run from the `Code/python/` directory:

```bash
python3 -m tests.timing.seconds.baseline_python_code
python3 -m tests.timing.seconds.baseline_c_code
python3 -m tests.timing.seconds.optimized_c_code
```

Results are saved in:

```text
../Results/timing_results/<YYYY-MM-DD>/
```
