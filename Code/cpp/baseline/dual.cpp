#include <torch/torch.h>
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <tuple>
#include <optional>
#include <vector>
#include <algorithm>
#include "linear.hpp"
#include "_cycle_counter.hpp"

namespace py = pybind11;

// Static dualization tables. perm[i] = which input blade feeds output blade i.
// sign[i] = sign factor for output blade i. These come from the PGA dual map
// and never change, so we keep them as compile-time constants.
static const int64_t DUAL_PERM[16] = {15, 14, 13, 12, 11, 10, 9, 8,
                                       7,  6,  5,  4,  3,  2, 1, 0};
static const float   DUAL_SIGN[16] = { 1.0f, -1.0f,  1.0f, -1.0f,
                                        1.0f,  1.0f, -1.0f,  1.0f,
                                        1.0f, -1.0f,  1.0f, -1.0f,
                                        1.0f, -1.0f,  1.0f,  1.0f};

/**
 * Builds the permutation and sign tensors used for dualization.
 * Kept for API compatibility — equi_dual itself no longer calls this.
 */
std::tuple<torch::Tensor, torch::Tensor> _compute_dualization(
    const torch::Device& device,
    torch::ScalarType dtype
) {
    int64_t* perm_arr = new int64_t[16];
    float*   sign_arr = new float[16];
    for (int i = 0; i < 16; ++i) {
        perm_arr[i] = DUAL_PERM[i];
        sign_arr[i] = DUAL_SIGN[i];
    }

    torch::Tensor perm = torch::from_blob(
        perm_arr, {16}, torch::dtype(torch::kInt64)
    ).clone();
    torch::Tensor sign = torch::from_blob(
        sign_arr, {16}, torch::dtype(torch::kFloat32)
    ).clone();

    delete[] perm_arr;
    delete[] sign_arr;

    return std::make_tuple(perm.to(device), sign.to(device, dtype));
}

/**
 * Applies the equivariant dual operation to the last dimension of x.
 * Raw implementation: out[b, i] = sign[i] * x[b, perm[i]].
 */
torch::Tensor equi_dual(const torch::Tensor& x) {
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) == 16,
                "equi_dual: last dimension must be 16");

    CYCLE_SCOPE("equi_dual");

    const auto out_device = x.device();
    const auto out_dtype  = x.scalar_type();

    auto x_cpu = x.contiguous().to(torch::kCPU, torch::kFloat32);
    const float* x_ptr = x_cpu.data_ptr<float>();

    int64_t batch_size = x_cpu.numel() / 16;
    float* out_ptr = new float[batch_size * 16];

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* x_row   = x_ptr   + b * 16;
        float*       out_row = out_ptr + b * 16;
        for (int i = 0; i < 16; ++i) {
            out_row[i] = DUAL_SIGN[i] * x_row[DUAL_PERM[i]];
        }
    }

    torch::Tensor result = torch::from_blob(
        out_ptr, x_cpu.sizes(), torch::dtype(torch::kFloat32)
    ).clone();
    delete[] out_ptr;

    return result.to(out_device, out_dtype);
}

/**
 * Precomputes the kernel tensor used to evaluate the equivariant join.
 * Layout: kernel_raw[k * 256 + i * 16 + j] == kernel[k, i, j].
 */
torch::Tensor _compute_efficient_join_kernel(
    const torch::Device& device,
    torch::ScalarType dtype
) {
    float* kernel_raw = new float[16 * 16 * 16]();

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            torch::Tensor x = torch::zeros(
                {16}, torch::dtype(dtype).device(device));
            torch::Tensor y = torch::zeros(
                {16}, torch::dtype(dtype).device(device));
            x.index_put_({i}, 1.0);
            y.index_put_({j}, 1.0);

            torch::Tensor result = equi_dual(
                outer_product(equi_dual(x), equi_dual(y))
            ).to(torch::kCPU, torch::kFloat32).contiguous();

            const float* res_ptr = result.data_ptr<float>();
            for (int k = 0; k < 16; ++k) {
                kernel_raw[k * 256 + i * 16 + j] = res_ptr[k];
            }
        }
    }

    torch::Tensor kernel = torch::from_blob(
        kernel_raw, {16, 16, 16}, torch::dtype(torch::kFloat32)
    ).clone();
    delete[] kernel_raw;

    return kernel.to(device, dtype);
}

/**
 * Computes the equivariant join of x and y using the precomputed kernel.
 *
 * Hot kernel: out[b, i] = sum_{j,k} kernel[i,j,k] * x[b,j] * y[b,k].
 * Reference scaling (when provided): out[..., :] *= reference[..., 14],
 * with full NumPy-style broadcasting between out's leading dims and ref's
 * leading dims, all done with raw pointer arithmetic.
 */
torch::Tensor equi_join(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const std::optional<torch::Tensor>& reference
) {
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) == 16,
                "equi_join: x last dimension must be 16");
    TORCH_CHECK(y.dim() >= 1 && y.size(-1) == 16,
                "equi_join: y last dimension must be 16");
    TORCH_CHECK(x.sizes() == y.sizes(),
                "equi_join: x and y must have matching shapes");

    CYCLE_SCOPE("equi_join");

    const auto out_device = x.device();
    const auto out_dtype  = x.scalar_type();

    // Build the join kernel once on CPU/float32.
    torch::Tensor kernel = _compute_efficient_join_kernel(
        torch::kCPU, torch::kFloat32
    ).contiguous();
    const float* kernel_ptr = kernel.data_ptr<float>();

    auto x_cpu = x.contiguous().to(torch::kCPU, torch::kFloat32);
    auto y_cpu = y.contiguous().to(torch::kCPU, torch::kFloat32);
    const float* x_ptr = x_cpu.data_ptr<float>();
    const float* y_ptr = y_cpu.data_ptr<float>();

    const std::vector<int64_t> x_shape = x_cpu.sizes().vec();  // e.g. [B, T, C, 16]
    const int64_t batch_size = x_cpu.numel() / 16;

    float* out_ptr = new float[batch_size * 16]();

    // ---- Hot loop: kernel evaluation in raw C ----
    // out[b, i] = sum_{j, k} kernel[i, j, k] * x[b, j] * y[b, k]
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* x_row = x_ptr + b * 16;
        const float* y_row = y_ptr + b * 16;
        float*       o_row = out_ptr + b * 16;

        for (int i = 0; i < 16; ++i) {
            const float* k_slice = kernel_ptr + i * 256;  // kernel[i, :, :]
            float sum = 0.0f;
            for (int j = 0; j < 16; ++j) {
                float xj = x_row[j];
                if (xj == 0.0f) continue;
                const float* k_row = k_slice + j * 16;     // kernel[i, j, :]
                for (int k = 0; k < 16; ++k) {
                    sum += k_row[k] * xj * y_row[k];
                }
            }
            o_row[i] = sum;
        }
    }

    // ---- Reference broadcast in raw C ----
    // Equivalent to Python's `ret * reference[..., [14]]`: take blade 14
    // (e_123) from reference and broadcast-multiply against the (..., 16)
    // result. We implement NumPy/PyTorch broadcasting manually:
    //   1. Right-align the leading-dim shapes (everything except the last 16).
    //   2. Each pair of dims must be equal, or one of them must be 1.
    //   3. A size-1 dim contributes stride 0 on the ref side, so different
    //      output positions read the same ref element.
    if (reference.has_value()) {
        auto ref_cpu = reference.value().contiguous()
                          .to(torch::kCPU, torch::kFloat32);
        TORCH_CHECK(ref_cpu.dim() >= 1 && ref_cpu.size(-1) == 16,
                    "equi_join: reference last dim must be 16");

        const std::vector<int64_t> ref_shape = ref_cpu.sizes().vec();
        const float* ref_ptr = ref_cpu.data_ptr<float>();

        // "Scalar grid" = leading dims, i.e. shape with the trailing 16 dropped.
        const int64_t out_rank = (int64_t)x_shape.size() - 1;
        const int64_t ref_rank = (int64_t)ref_shape.size() - 1;
        const int64_t bcast_rank = std::max(out_rank, ref_rank);

        // Right-align: pad shorter shape with 1s on the left.
        std::vector<int64_t> out_dims(bcast_rank, 1);
        std::vector<int64_t> ref_dims(bcast_rank, 1);
        for (int64_t d = 0; d < out_rank; ++d) {
            out_dims[bcast_rank - out_rank + d] = x_shape[d];
        }
        for (int64_t d = 0; d < ref_rank; ++d) {
            ref_dims[bcast_rank - ref_rank + d] = ref_shape[d];
        }

        // Validate broadcastability.
        for (int64_t d = 0; d < bcast_rank; ++d) {
            int64_t a = out_dims[d], b = ref_dims[d];
            TORCH_CHECK(a == b || a == 1 || b == 1,
                        "equi_join: reference shape not broadcastable to result");
        }

        // Compute ref strides over its scalar grid (i.e. in float units,
        // stepping by 16 per scalar-grid element since ref is contiguous
        // and its last dim is size 16). A size-1 dim broadcasts by getting
        // stride 0, so any output position along that dim reads the same
        // ref element.
        std::vector<int64_t> ref_strides(bcast_rank, 0);
        {
            int64_t s = 16;
            for (int64_t d = ref_rank - 1; d >= 0; --d) {
                int64_t aligned_d = bcast_rank - ref_rank + d;
                ref_strides[aligned_d] =
                    (ref_dims[aligned_d] == 1) ? 0 : s;
                s *= ref_dims[aligned_d];
            }
            // Left-padded dims (where ref_rank < bcast_rank) keep stride 0.
        }

        // Walk every scalar-grid position b and multiply that row of out
        // by ref[..., 14] at the matching (broadcasted) position.
        for (int64_t b = 0; b < batch_size; ++b) {
            // Decode b into a multi-index over out_dims (last-dim-fastest).
            int64_t rem = b;
            int64_t ref_offset = 0;
            for (int64_t d = bcast_rank - 1; d >= 0; --d) {
                int64_t coord = rem % out_dims[d];
                rem /= out_dims[d];
                ref_offset += coord * ref_strides[d];
            }
            // ref_offset addresses ref[..., 0] for this position; +14
            // selects the e_123 blade.
            float scale = ref_ptr[ref_offset + 14];

            float* o_row = out_ptr + b * 16;
            for (int i = 0; i < 16; ++i) {
                o_row[i] *= scale;
            }
        }
    }

    torch::Tensor result = torch::from_blob(
        out_ptr, x_cpu.sizes(), torch::dtype(torch::kFloat32)
    ).clone();
    delete[] out_ptr;

    return result.to(out_device, out_dtype);
}

PYBIND11_MODULE(dual, m) {
    m.def("outer_product", &outer_product, "outer_product");
    m.def("geometric_product", &geometric_product, "geometric_product");
    m.def("equi_dual", &equi_dual, "equi_dual");
    m.def("equi_join", &equi_join, "equi_join",
          py::arg("x"), py::arg("y"), py::arg("reference") = std::nullopt);
    BIND_CYCLE_COUNTER(m);
}