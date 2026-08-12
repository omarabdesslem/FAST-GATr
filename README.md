The geometric algebra transformer (GATr) is a neural network architecture designed to process geometric data, such as planes,  vectors, and 3D points. For this project, we view implementing GATr primarily as a structured computation problem by abstracting away the underlying mathematics. 

![EzGATr overview](Notes/Images/ezgatr_overview.png)


We plan to optimize the following modules and functions: 

- <span style="color:#1f77b4">`EquiLinear`</span>: equivariant linear layer that maps input multivectors to output multivectors while preserving geometric structure.
- <span style="color:#ff7f0e">`EquiRMSNorm`</span>: RMS normalization layer adapted for equivariant multivector features.
- <span style="color:#2ca02c">`geometric_product`</span>: core geometric algebra operation combining two multivectors into another multivector.
- <span style="color:#d62728">`equi_join`</span>: equivariant join operation used to combine geometric entities such as points, lines, and planes.
- <span style="color:#9467bd">`equi_geometric_attention`</span>: attention mechanism that uses geometric algebra operations to compute geometry-aware interactions.
- <span style="color:#17becf">`scaler_gated_gelu`</span>: gated GELU activation applied to scalar channels to control nonlinear feature updates.

We use the following paper for reference:

Brehmer, J., De Haan, P., Behrends, S. and Cohen, T.S., 2023. Geometric algebra
transformer. Advances in Neural Information Processing Systems, 36, pp.35472-35496

Pages 3-7 and Appendix B define the core operations (geometric product, equivariant linear maps, geometric attention, join, gated nonlinearity, and normalization) that we optimize in this project.


As a starting point we use the existing EzGATr (Easy Geometric Algebra Transformer) Python PyTorch code.


We'll be creating a straightforward implementation in C/C++

![Performance Results](Results/timing_results/Plots/comparison_plots/M1/end_to_end/end_to_end_with_baseline_log.png)


## Build Output

The compiled C++ extension modules are built into:

```text
Code/python/ops/
```

Temporary compiler files are stored in:

```text
Code/python/build/
```

The `Code/python/build/` folder and compiled `.so` files should not be committed.

Generated experiment artifacts are kept outside `Code/`:

```text
Results/timing_results/
Results/profiling/
Results/disassembly/
Results/logs/
```

## Results

![Baseline vs FASTGATr](Results/timing_results/Plots/comparison_plots/M1/speedups/baseline_cpp_vs_simd_cpp_end_to_end_speedup_ratio_log.png)
![State Of The Art vs FASTGATr](Results/timing_results/Plots/comparison_plots/M1/speedups/python_vs_simd_cpp_end_to_end_speedup_ratio_log.png)
