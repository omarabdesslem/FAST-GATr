#include <torch/extension.h>
#include <cmath>
#include <optional>
#include <pybind11/pybind11.h>
#include "_cycle_counter.hpp"

// Blade indices that don't contain e_0 (scalar, e_i, e_ij, e_123).
// Matches _compute_inner_product_selector in the python.
static constexpr int IP_IDX[8] = {0, 2, 3, 4, 8, 9, 10, 14};


// x, out are [N][C][16]; weight is [C] (or null to skip rescale).
// N collapses everything before the channel dim so we only need one loop
// over the leading shape.
void equi_rms_norm_kernel(
    float* out,
    const float* x,
    const float* weight,
    float eps,
    int N, int C)
{
    for (int n = 0; n < N; n++) {
        const float* x_n = x + n * C * 16;
        float* out_n = out + n * C * 16;

        // Sum the per-channel inner products, then take the mean.
        // The python does mean(inner_product(x,x), dim=-2) which is the same thing
        // since every channel contributes one scalar.
        float sum_inner = 0.0f;
        for (int c = 0; c < C; c++) {
            const float* x_nc = x_n + c * 16;
            float ip = 0.0f;
            for (int s = 0; s < 8; s++) {
                float v = x_nc[IP_IDX[s]];
                ip += v * v;
            }
            sum_inner += ip;
        }
        float mean = sum_inner / (float)C;
        if (mean < eps) mean = eps;
        float scale = 1.0f / std::sqrt(mean);

        // Apply the shared scale and fold the per-channel weight into it.
        for (int c = 0; c < C; c++) {
            const float* x_nc = x_n + c * 16;
            float* out_nc = out_n + c * 16;
            float s = scale * ((weight != nullptr) ? weight[c] : 1.0f);
            for (int k = 0; k < 16; k++) {
                out_nc[k] = x_nc[k] * s;
            }
        }
    }
}


// x, out are [M][16]. M collapses every leading dim; only the last (blade)
// dim matters here since the gate is taken from the scalar blade.
void scaler_gated_gelu_kernel(
    float* out,
    const float* x,
    int M)
{
    // tanh approximation of GELU, same as torch's approximate="tanh".
    const float k0 = 0.7978845608028654f;  // sqrt(2/pi)
    const float k1 = 0.044715f;

    for (int m = 0; m < M; m++) {
        const float* x_m = x + m * 16;
        float* out_m = out + m * 16;

        float v = x_m[0];
        float gate = 0.5f * v * (1.0f + std::tanh(k0 * (v + k1 * v * v * v)));

        for (int k = 0; k < 16; k++) {
            out_m[k] = x_m[k] * gate;
        }
    }
}


// x: (..., C, 16), weight: (C,) or None
torch::Tensor asl_equi_rms_norm(
    torch::Tensor x,
    std::optional<torch::Tensor> weight = std::nullopt,
    std::optional<double> eps = std::nullopt)
{
    CYCLE_SCOPE("equi_rms_norm");
    x = x.contiguous();
    TORCH_CHECK(x.dim() >= 2, "equi_rms_norm: x must have shape (..., C, 16)");
    TORCH_CHECK(x.size(-1) == 16, "equi_rms_norm: last dim must be 16");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "equi_rms_norm: float32 only");

    int C = (int)x.size(-2);
    int N = (int)(x.numel() / (C * 16));

    float eps_f = eps.has_value()
        ? (float)eps.value()
        : std::numeric_limits<float>::epsilon();

    const float* w_ptr = nullptr;
    torch::Tensor w_contig;
    if (weight.has_value()) {
        w_contig = weight.value().contiguous();
        TORCH_CHECK(w_contig.scalar_type() == torch::kFloat32, "equi_rms_norm: weight float32 only");
        TORCH_CHECK((int)w_contig.numel() == C, "equi_rms_norm: weight size must equal C");
        w_ptr = w_contig.data_ptr<float>();
    }

    auto out = torch::empty_like(x);
    equi_rms_norm_kernel(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        w_ptr,
        eps_f,
        N, C);
    return out;
}


// x: (..., 16)
torch::Tensor asl_scaler_gated_gelu(torch::Tensor x) {
    CYCLE_SCOPE("scaler_gated_gelu");
    x = x.contiguous();
    TORCH_CHECK(x.dim() >= 1, "scaler_gated_gelu: x must have shape (..., 16)");
    TORCH_CHECK(x.size(-1) == 16, "scaler_gated_gelu: last dim must be 16");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "scaler_gated_gelu: float32 only");

    int M = (int)(x.numel() / 16);

    auto out = torch::empty_like(x);
    scaler_gated_gelu_kernel(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        M);
    return out;
}


PYBIND11_MODULE(norm, m) {
    m.def("equi_rms_norm", &asl_equi_rms_norm, "PGA inner-induced RMS norm",
        pybind11::arg("x"),
        pybind11::arg("weight") = std::nullopt,
        pybind11::arg("eps") = std::nullopt);
    m.def("scaler_gated_gelu", &asl_scaler_gated_gelu, "Scalar-gated GELU (tanh approx)",
        pybind11::arg("x"));
    BIND_CYCLE_COUNTER(m);
}
