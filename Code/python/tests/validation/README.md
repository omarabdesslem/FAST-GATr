# Validation Tests

This folder contains validation tests for comparing the original Python EzGATr implementation with the ASL C++ implementation.

## Purpose

The validation tests check that the C++ implementation produces results consistent with the Python implementation.

They are intended to verify correctness after:

- compiling or modifying the C++ extensions
- regenerating basis files
- changing Python reference code
- updating tensor, algebra, or model logic

## Structure

```text
tests/validation/
├── baseline/
│   ├── baseline_test_end_to_end.py
│   ├── baseline_test_equi_join.py
│   ├── baseline_test_equilinear.py
│   ├── baseline_test_geometric_attention.py
│   └── baseline_test_geometric_product.py
├── optimized/
│   ├── optimized_test_end_to_end.py
│   ├── optimized_test_equi_join.py
│   ├── optimized_test_equilinear.py
│   ├── optimized_test_geometric_attention.py
│   └── optimized_test_geometric_product.py
├── optimized_for_cache/
│   ├── optimized_for_cache_test_end_to_end.py
│   ├── optimized_for_cache_test_equi_join.py
│   ├── optimized_for_cache_test_equilinear.py
│   ├── optimized_for_cache_test_geometric_attention.py
│   └── optimized_for_cache_test_geometric_product.py
└── README.md
```

## Run

From the `Code/python/` directory:

```bash
python3 -m tests.validation.baseline.baseline_test_end_to_end
python3 -m tests.validation.optimized.optimized_test_end_to_end
python3 -m tests.validation.optimized_for_cache.optimized_for_cache_test_end_to_end
```

Each validation test writes CSV results to:

```text
../Results/validation/<variant>/<YYYY-MM-DD>/
```

The `baseline/` and `optimized/` tests use the same Python reference checks.
Each test rebuilds the matching C++ implementation before importing `ops/`:

- `baseline/` tests build from `cpp/baseline`
- `optimized/` tests build from `cpp/optimized`
- `optimized_for_cache/` tests build from `cpp/optimized_for_cache`

This avoids accidentally validating whichever extension was last built into
`ops/`.
