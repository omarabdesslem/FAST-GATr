#include <torch/extension.h>
#include <map>
#include <tuple>
#include <torch/script.h>
#include <cstring>
#include <vector>
#include <arm_neon.h>
#include "_cycle_counter.hpp"
#include "../generated_kernels/gp_kernel_neon.cpp"
#include "../generated_kernels/gp_kernel.cpp"

// Input:
//   x, y: two tensors with the same shape. The last dimension must be 16.
// Output:
//   A tensor with the same shape as x. Each 16-value row is x * y.
// Why:
//   This is the C++ implementation of the geometric product. It uses the
//   faster generated kernel for groups of 4 rows, then handles any leftover
//   rows one by one.
torch::Tensor geometric_product(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.sizes() == y.sizes(), "geometric_product: x and y must have identical shapes");
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) == 16, "geometric_product: last dimension must be 16");

    CYCLE_SCOPE("geometric_product");

    const auto out_device = x.device();
    const auto out_dtype = x.scalar_type();

    // Convert once to CPU float32. Calling contiguous first would create an
    // extra temporary tensor in some cases.
    auto x_cpu = x.to(torch::kCPU, torch::kFloat32);
    auto y_cpu = y.to(torch::kCPU, torch::kFloat32);
    const float* x_ptr = x_cpu.data_ptr<float>();
    const float* y_ptr = y_cpu.data_ptr<float>();

    int64_t batch_size = x_cpu.numel() / 16;
    auto result_cpu = torch::empty_like(x_cpu);
    float* result_ptr = result_cpu.data_ptr<float>();

    // Process 4 rows at a time, then handle the remaining rows.
    int64_t b = 0;
    for (; b + 4 <= batch_size; b += 4) {
        const int64_t b16 = b * 16;
        gp_kernel_four_batch(x_ptr + b16, y_ptr + b16, result_ptr + b16);
    }
    for (; b < batch_size; ++b) {
        const int64_t b16 = b * 16;
        gp_kernel_one_batch(x_ptr + b16, y_ptr + b16, result_ptr + b16);
    }

    return result_cpu.to(out_device, out_dtype);
}

// Input:
//   x:      [B, T, channels_in, 16]
//   weight: [channels_out, channels_in, 9]
//   basis:  [9, 16, 16], passed in to keep the same API, but not read here.
//   bias:   [channels_out] or nullptr
// Output:
//   out:    [B, T, channels_out, 16]
// Why:
//   This is the normal EquiLinear path. It computes the same result as the
//   original code, but uses NEON for the blade groups that are easy to process
//   together.
void equi_linear_kernel(float* __restrict out, const float* __restrict x, const float* __restrict weight,
    const float* /*basis*/, const float* __restrict bias,
    int B, int T, int channels_in, int channels_out) {

    const float C1 = 1.0f;                  // w = 0,4,5,8
    const float C2 = 0.5f;                  // w = 1,3
    const float C3 = 0.57735026918962576f;  // w = 6,7   (1/sqrt(3))
    const float C6 = 0.40824829046386301f;  // w = 2      (1/sqrt(6))

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const int bt = b * T + t;
            const float* __restrict x_bt = x + bt * channels_in * 16;
            float* __restrict out_bt = out + bt * channels_out * 16;

            for (int o = 0; o < channels_out; o++) {
                const float* __restrict w_o = weight + o * channels_in * 9;
                float* __restrict out_o = out_bt + o * 16;

                float a0 = 0.f;
                float a1_off = 0.f;
                float a5_off = 0.f, a6_off = 0.f, a7_off = 0.f;
                float a9 = 0.f, a10 = 0.f;
                float a11_off = 0.f, a12_off = 0.f, a13_off = 0.f;
                float a15_diag = 0.f, a15_off = 0.f;

                // These blade groups can be updated 4 values at a time.
                float32x4_t acc1_4 = vdupq_n_f32(0.0f);
                float32x4_t acc5_8 = vdupq_n_f32(0.0f);
                float32x4_t acc11_14 = vdupq_n_f32(0.0f);

                for (int i = 0; i < channels_in; i++) {
                    const float* __restrict wi = w_o + i * 9;
                    const float* __restrict xi = x_bt + i * 16;

                    const float w0 = wi[0] * C1;
                    const float w1 = wi[1] * C2;
                    const float w2 = wi[2] * C6;
                    const float w3 = wi[3] * C2;
                    const float w4 = wi[4] * C1;
                    const float w5 = wi[5] * C1;
                    const float w6 = wi[6] * C3;
                    const float w7 = wi[7] * C3;
                    const float w8 = wi[8] * C1;

                    // Main blade contributions.
                    a0       += w0 * xi[0];
                    acc1_4    = vfmaq_n_f32(acc1_4, vld1q_f32(xi + 1), w1);
                    acc5_8    = vfmaq_n_f32(acc5_8, vld1q_f32(xi + 5), w2);
                    a9       += w2 * xi[9];
                    a10      += w2 * xi[10];
                    acc11_14  = vfmaq_n_f32(acc11_14, vld1q_f32(xi + 11), w3);
                    a15_diag += w4 * xi[15];

                    // Extra contributions that go to different output blades.
                    a1_off  += w5 * xi[0];   // (1, 0)
                    a5_off  += w6 * xi[2];   // (5, 2)
                    a6_off  += w6 * xi[3];   // (6, 3)
                    a7_off  += w6 * xi[4];   // (7, 4)
                    a11_off += w7 * xi[8];   // (11, 8)
                    a12_off += w7 * xi[9];   // (12, 9)
                    a13_off += w7 * xi[10];  // (13, 10)
                    a15_off += w8 * xi[14];  // (15, 14)
                }

                const float a1_diag = vgetq_lane_f32(acc1_4, 0);
                const float a2 = vgetq_lane_f32(acc1_4, 1);
                const float a3 = vgetq_lane_f32(acc1_4, 2);
                const float a4 = vgetq_lane_f32(acc1_4, 3);
                const float a5_diag = vgetq_lane_f32(acc5_8, 0);
                const float a6_diag = vgetq_lane_f32(acc5_8, 1);
                const float a7_diag = vgetq_lane_f32(acc5_8, 2);
                const float a8 = vgetq_lane_f32(acc5_8, 3);
                const float a11_diag = vgetq_lane_f32(acc11_14, 0);
                const float a12_diag = vgetq_lane_f32(acc11_14, 1);
                const float a13_diag = vgetq_lane_f32(acc11_14, 2);
                const float a14 = vgetq_lane_f32(acc11_14, 3);

                out_o[0]  = a0;
                out_o[1]  = a1_diag  + a1_off;
                out_o[2]  = a2;
                out_o[3]  = a3;
                out_o[4]  = a4;
                out_o[5]  = a5_diag  + a5_off;
                out_o[6]  = a6_diag  + a6_off;
                out_o[7]  = a7_diag  + a7_off;
                out_o[8]  = a8;
                out_o[9]  = a9;
                out_o[10] = a10;
                out_o[11] = a11_diag + a11_off;
                out_o[12] = a12_diag + a12_off;
                out_o[13] = a13_diag + a13_off;
                out_o[14] = a14;
                out_o[15] = a15_diag + a15_off;

                if (bias != NULL) {
                    out_o[0] += bias[o];
                }
            }
        }
    }
}

// Input:
//   weight: [channels_out, channels_in, 9]
// Output:
//   packed: [channels_in, channels_out, 9]
// Why:
//   Reorders the weights into the layout used by the faster kernel. It also
//   applies the fixed scale factors once here, so the forward pass does less
//   repeated work.
torch::Tensor pack_equi_linear_weight(torch::Tensor weight) {
    TORCH_CHECK(weight.dim() == 3 && weight.size(2) == 9,
                "pack_equi_linear_weight: weight must have shape [out, in, 9]");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32,
                "pack_equi_linear_weight: float32 only");

    weight = weight.contiguous();
    const int channels_out = (int)weight.size(0);
    const int channels_in = (int)weight.size(1);

    auto packed = torch::empty({channels_in, channels_out, 9}, weight.options());
    const float* __restrict weight_ptr = weight.data_ptr<float>();
    float* __restrict packed_ptr = packed.data_ptr<float>();

    const float C1 = 1.0f;
    const float C2 = 0.5f;
    const float C3 = 0.57735026918962576f;
    const float C6 = 0.40824829046386301f;

    for (int i = 0; i < channels_in; ++i) {
        for (int o = 0; o < channels_out; ++o) {
            const float* __restrict weight_oi = weight_ptr + (o * channels_in + i) * 9;
            float* __restrict packed_io = packed_ptr + (i * channels_out + o) * 9;
            packed_io[0] = weight_oi[0] * C1;
            packed_io[1] = weight_oi[1] * C2;
            packed_io[2] = weight_oi[2] * C6;
            packed_io[3] = weight_oi[3] * C2;
            packed_io[4] = weight_oi[4] * C1;
            packed_io[5] = weight_oi[5] * C1;
            packed_io[6] = weight_oi[6] * C3;
            packed_io[7] = weight_oi[7] * C3;
            packed_io[8] = weight_oi[8] * C1;
        }
    }
    return packed;
}

// Input:
//   The temporary sums for one output channel.
// Output:
//   out_channel[0..15], the final 16 values for that output channel.
// Why:
//   Keeps the final write step in one place, so the packed kernel is easier to
//   read.
static inline void write_equi_linear_row(
    float* __restrict out_channel,
    float scalar_blade,
    float32x4_t blades_1_4,
    float32x4_t blades_5_8,
    float blade_9,
    float blade_10,
    float32x4_t blades_11_14,
    float blade_15_diag,
    float blade_1_off,
    float blade_5_off,
    float blade_6_off,
    float blade_7_off,
    float blade_11_off,
    float blade_12_off,
    float blade_13_off,
    float blade_15_off,
    float bias)
{
    out_channel[0]  = scalar_blade + bias;
    out_channel[1]  = vgetq_lane_f32(blades_1_4, 0) + blade_1_off;
    out_channel[2]  = vgetq_lane_f32(blades_1_4, 1);
    out_channel[3]  = vgetq_lane_f32(blades_1_4, 2);
    out_channel[4]  = vgetq_lane_f32(blades_1_4, 3);
    out_channel[5]  = vgetq_lane_f32(blades_5_8, 0) + blade_5_off;
    out_channel[6]  = vgetq_lane_f32(blades_5_8, 1) + blade_6_off;
    out_channel[7]  = vgetq_lane_f32(blades_5_8, 2) + blade_7_off;
    out_channel[8]  = vgetq_lane_f32(blades_5_8, 3);
    out_channel[9]  = blade_9;
    out_channel[10] = blade_10;
    out_channel[11] = vgetq_lane_f32(blades_11_14, 0) + blade_11_off;
    out_channel[12] = vgetq_lane_f32(blades_11_14, 1) + blade_12_off;
    out_channel[13] = vgetq_lane_f32(blades_11_14, 2) + blade_13_off;
    out_channel[14] = vgetq_lane_f32(blades_11_14, 3);
    out_channel[15] = blade_15_diag + blade_15_off;
}

// Input:
//   x:             [B, T, channels_in, 16]
//   packed_weight: [channels_in, channels_out, 9]
//   bias:          [channels_out] or nullptr
// Output:
//   out:           [B, T, channels_out, 16]
// Why:
//   This is the faster EquiLinear path. It expects weights that were already
//   packed by pack_equi_linear_weight, then computes the output using the same
//   math with less memory shuffling.
void equi_linear_packed_kernel(float* __restrict out, const float* __restrict x,
    const float* __restrict packed_weight, const float* __restrict bias,
    int B, int T, int channels_in, int channels_out) {

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const int batch_token = b * T + t;
            const float* __restrict x_row = x + batch_token * channels_in * 16;
            float* __restrict out_row = out + batch_token * channels_out * 16;

            for (int o = 0; o < channels_out; ++o) {
                float scalar_blade = 0.f;
                float blade_1_off = 0.f;
                float blade_5_off = 0.f, blade_6_off = 0.f, blade_7_off = 0.f;
                float blade_9 = 0.f, blade_10 = 0.f;
                float blade_11_off = 0.f, blade_12_off = 0.f, blade_13_off = 0.f;
                float blade_15_diag = 0.f, blade_15_off = 0.f;
                float32x4_t blades_1_4 = vdupq_n_f32(0.0f);
                float32x4_t blades_5_8 = vdupq_n_f32(0.0f);
                float32x4_t blades_11_14 = vdupq_n_f32(0.0f);

                for (int i = 0; i < channels_in; i++) {
                    const float* __restrict x_channel = x_row + i * 16;
                    const float* __restrict packed_io = packed_weight + (i * channels_out + o) * 9;
                    scalar_blade += packed_io[0] * x_channel[0];
                    blades_1_4 = vfmaq_n_f32(blades_1_4, vld1q_f32(x_channel + 1), packed_io[1]);
                    blades_5_8 = vfmaq_n_f32(blades_5_8, vld1q_f32(x_channel + 5), packed_io[2]);
                    blade_9 += packed_io[2] * x_channel[9];
                    blade_10 += packed_io[2] * x_channel[10];
                    blades_11_14 = vfmaq_n_f32(blades_11_14, vld1q_f32(x_channel + 11), packed_io[3]);
                    blade_15_diag += packed_io[4] * x_channel[15];
                    blade_1_off += packed_io[5] * x_channel[0];
                    blade_5_off += packed_io[6] * x_channel[2];
                    blade_6_off += packed_io[6] * x_channel[3];
                    blade_7_off += packed_io[6] * x_channel[4];
                    blade_11_off += packed_io[7] * x_channel[8];
                    blade_12_off += packed_io[7] * x_channel[9];
                    blade_13_off += packed_io[7] * x_channel[10];
                    blade_15_off += packed_io[8] * x_channel[14];
                }
                write_equi_linear_row(out_row + o * 16, scalar_blade, blades_1_4, blades_5_8,
                    blade_9, blade_10, blades_11_14, blade_15_diag, blade_1_off,
                    blade_5_off, blade_6_off, blade_7_off, blade_11_off, blade_12_off,
                    blade_13_off, blade_15_off, bias ? bias[o] : 0.0f);
            }
        }
    }
}

// Input:
//   x:             [B, T, channels_in, 16]
//   packed_weight: [channels_in, channels_out, 9]
//   basis:         passed in to match the normal function, but not read here.
//   bias:          optional [channels_out]
// Output:
//   [B, T, channels_out, 16]
// Why:
//   This is the Python-callable wrapper for the packed path. Python can pack
//   the weights once, cache them, and call this function many times.
torch::Tensor asl_equi_linear_packed(torch::Tensor x, torch::Tensor packed_weight, torch::Tensor basis,
    std::optional<torch::Tensor> bias = std::nullopt) {
    CYCLE_SCOPE("equi_linear");
    x = x.contiguous();
    packed_weight = packed_weight.contiguous();
    basis = basis.contiguous();
    TORCH_CHECK(packed_weight.dim() == 3 && packed_weight.size(2) == 9,
                "equi_linear_packed: packed weight must have shape [in, out, 9]");

    int B = x.size(0);
    int T = x.size(1);
    int channels_in = x.size(2);
    int channels_out = packed_weight.size(1);
    TORCH_CHECK(packed_weight.size(0) == channels_in,
                "equi_linear_packed: packed weight input channels must match x");
    auto out = torch::empty({B, T, channels_out, 16}, x.options());

    const float* bias_ptr = nullptr;
    torch::Tensor bias_contig;
    if (bias.has_value()) {
        bias_contig = bias.value().contiguous();
        bias_ptr = bias_contig.data_ptr<float>();
    }

    equi_linear_packed_kernel(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        packed_weight.data_ptr<float>(),
        bias_ptr,
        B, T, channels_in, channels_out
    );
    return out;
}
// Input:
//   x:      [B, T, channels_in, 16]
//   weight: [channels_out, channels_in, 9]
//   basis:  [9, 16, 16]
//   bias:   optional [channels_out]
// Output:
//   [B, T, channels_out, 16]
// Why:
//   This is the Python-callable wrapper for the normal path. It is still useful
//   when weights are not packed or when we want the simpler fallback behavior.
torch::Tensor asl_equi_linear(torch::Tensor x, torch::Tensor weight, torch::Tensor basis,
    std::optional<torch::Tensor> bias = std::nullopt) {
    CYCLE_SCOPE("equi_linear");
    x = x.contiguous();
    weight = weight.contiguous();
    basis = basis.contiguous();
    
    // Create the output with the expected batch/token/channel shape.
    int B = x.size(0);
    int T = x.size(1);
    int channels_in = x.size(2);
    int channels_out = weight.size(0);
    auto out = torch::empty({B, T, channels_out, 16}, x.options());
    
    // Bias is optional, so pass nullptr when it is missing.
    const float* bias_ptr = nullptr;
    torch::Tensor bias_contig;
    if (bias.has_value()) {
        bias_contig = bias.value().contiguous();
        bias_ptr = bias_contig.data_ptr<float>();
    }

    equi_linear_kernel(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        basis.data_ptr<float>(),
        bias_ptr,
        B, T, channels_in, channels_out
    );

    return out;
}

// Input:
//   Python module object created by pybind11.
// Output:
//   Makes these C++ functions available from Python.
// Why:
//   Python needs these bindings to call the normal path, the packed path, and
//   the weight-packing helper.
#include <pybind11/pybind11.h>

PYBIND11_MODULE(linear, m) {
    m.def("geometric_product", &geometric_product, "Geometric product");
    m.def("pack_equi_linear_weight", &pack_equi_linear_weight, "Pack and pre-scale equi_linear weights");
    m.def("equi_linear_packed", &asl_equi_linear_packed, "pin-equivariant linear map with packed weights",
        pybind11::arg("x"), pybind11::arg("packed_weight"), pybind11::arg("basis"), pybind11::arg("bias") = std::nullopt);
    m.def("equi_linear", &asl_equi_linear, "pin-equivariant linear map",
        pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("basis"), pybind11::arg("bias") = std::nullopt);
    BIND_CYCLE_COUNTER(m);
}
