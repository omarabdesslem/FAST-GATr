#include <torch/extension.h>
#include <cmath>
#include <optional>
#include <pybind11/pybind11.h>
#include <arm_neon.h>
#include "_cycle_counter.hpp"

// The PGA inner product selector is sparse: only these 8 of the 16 blades
// contribute to ||x||. Hardcoding the selected blades avoids the selector
// array lookup and the tiny loop around it in the normalization hot path.
static inline float selected_inner_product_square(const float* __restrict x) {
    const float32x4_t x0_3 = vld1q_f32(x);
    const float32x4_t x8_11 = vld1q_f32(x + 8);
    const float32x4_t sq0_3 = vmulq_f32(x0_3, x0_3);
    const float32x4_t sq8_11 = vmulq_f32(x8_11, x8_11);

    float sum = vaddvq_f32(sq0_3) - vgetq_lane_f32(sq0_3, 1);
    sum += x[4] * x[4];
    sum += vaddvq_f32(sq8_11) - vgetq_lane_f32(sq8_11, 3);
    sum += x[14] * x[14];
    return sum;
}

static inline void scale_blades_16(
    float* __restrict out,
    const float* __restrict x,
    float scale)
{
    const float32x4_t s = vdupq_n_f32(scale);
    vst1q_f32(out,      vmulq_f32(vld1q_f32(x),      s));
    vst1q_f32(out + 4,  vmulq_f32(vld1q_f32(x + 4),  s));
    vst1q_f32(out + 8,  vmulq_f32(vld1q_f32(x + 8),  s));
    vst1q_f32(out + 12, vmulq_f32(vld1q_f32(x + 12), s));
}


// x, out are [N][C][16]; weight is [C] (or null to skip rescale).
// N collapses everything before the channel dim so we only need one loop
// over the leading shape.
void equi_rms_norm_kernel(
    float* __restrict out,
    const float* __restrict x,
    const float* __restrict weight,
    float eps,
    int N, int C)
{
    for (int n = 0; n < N; n++) {
        const float* __restrict x_n = x + n * C * 16;
        float* __restrict out_n = out + n * C * 16;

        // Sum the per-channel inner products, then take the mean.
        // The python does mean(inner_product(x,x), dim=-2) which is the same thing
        // since every channel contributes one scalar.
        float sum_inner = 0.0f;
        for (int c = 0; c < C; c++) {
            const float* __restrict x_nc = x_n + c * 16;
            sum_inner += selected_inner_product_square(x_nc);
        }
        float mean = sum_inner / (float)C;
        if (mean < eps) mean = eps;
        float scale = 1.0f / std::sqrt(mean);

        // Apply the shared scale and fold the per-channel weight into it.
        for (int c = 0; c < C; c++) {
            const float* __restrict x_nc = x_n + c * 16;
            float* __restrict out_nc = out_n + c * 16;
            float s = scale * ((weight != nullptr) ? weight[c] : 1.0f);
            scale_blades_16(out_nc, x_nc, s);
        }
    }
}


// x, out are [M][16]. M collapses every leading dim; only the last (blade)
// dim matters here since the gate is taken from the scalar blade.
void scaler_gated_gelu_kernel(
    float* __restrict out,
    const float* __restrict x,
    int M)
{
    // tanh approximation of GELU, same as torch's approximate="tanh".
    const float k0 = 0.7978845608028654f;  // sqrt(2/pi)
    const float k1 = 0.044715f;

    for (int m = 0; m < M; m++) {
        const float* __restrict x_m = x + m * 16;
        float* __restrict out_m = out + m * 16;

        float v = x_m[0];
        float v2 = v * v;
        float gate = 0.5f * v * (1.0f + std::tanh(k0 * (v + k1 * v * v2)));

        scale_blades_16(out_m, x_m, gate);
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
