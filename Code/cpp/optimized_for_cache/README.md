# Optimized C++ Kernels

This directory contains the C++ implementation used by `Code/python/setup.py`.

## `norm.cpp`

`norm.cpp` exports two Python-callable functions through `ops.norm`:

- `equi_rms_norm`
- `scaler_gated_gelu`

Each exported function has two layers:

- wrapper: `asl_equi_rms_norm` and `asl_scaler_gated_gelu`
- kernel: `equi_rms_norm_kernel` and `scaler_gated_gelu_kernel`


### `equi_rms_norm_kernel`

The equivariant RMS norm uses a sparse inner-product selector. Only 8 of the 16
multivector blades contribute to the norm:

```text
0, 2, 3, 4, 8, 9, 10, 14
```

The optimized kernel hardcodes those eight squared terms instead of looping over
an index table. This avoids the selector lookup and makes the fixed arithmetic
visible to the compiler.

The final per-blade rescale still applies to all 16 blades. That operation is
left as a simple loop because it is dense, obvious, and easy for the compiler to
unroll if profitable.
