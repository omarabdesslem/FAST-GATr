#include <torch/extension.h>
#include <map>
#include <tuple>
#include <torch/script.h>
#include <cstring>
#include <vector>
#include "_cycle_counter.hpp"
#include "../generated_kernels/gp_kernel.cpp"


torch::Tensor geometric_product(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.sizes() == y.sizes(), "geometric_product: x and y must have identical shapes");
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) == 16, "geometric_product: last dimension must be 16");

    CYCLE_SCOPE("geometric_product");

    const auto out_device = x.device();
    const auto out_dtype = x.scalar_type();

    auto x_cpu = x.contiguous().to(torch::kCPU, torch::kFloat32);
    auto y_cpu = y.contiguous().to(torch::kCPU, torch::kFloat32);
    const float* x_ptr = x_cpu.data_ptr<float>();
    const float* y_ptr = y_cpu.data_ptr<float>();

    int64_t batch_size = x_cpu.numel() / 16;
    auto result_cpu = torch::empty_like(x_cpu);
    float* result_ptr = result_cpu.data_ptr<float>();


    for (int64_t b = 0; b < batch_size; ++b) {
        int b16 = b * 16;
        gp_kernel_one_batch(x_ptr + b16, y_ptr + b16, result_ptr + b16);
    }
    

    return result_cpu.to(out_device, out_dtype);
}

// Equilinear function substitute
//   x:      [batch][tokens][channnels_in][16]
//   weight: [out_ch][channels_in][9]
//   basis:  [9][16][16]
//   out:    [batch][tokens][channels_out][16]
//   bias:   [out_ch]
//
// Basis is sparse and known at compile time, so we don't read it -- the
// non-zero (w, s) -> d wiring is hardcoded below. Each basis slice w has N
// ones; after normalization every entry is 1/sqrt(N). The four distinct
// values are folded into the weights once per (o, i).

void equi_linear_kernel(float* out, const float* x, const float* weight, const float* /*basis*/, const float* bias,
    int B, int T, int channels_in, int channels_out) {

    const float C1 = 1.0f;                  // w = 0,4,5,8
    const float C2 = 0.5f;                  // w = 1,3
    const float C3 = 0.57735026918962576f;  // w = 6,7   (1/sqrt(3))
    const float C6 = 0.40824829046386301f;  // w = 2      (1/sqrt(6))

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const int bt = b * T + t;
            const float* x_bt = x + bt * channels_in * 16;
            float* out_bt = out + bt * channels_out * 16;

            for (int o = 0; o < channels_out; o++) {
                const float* w_o = weight + o * channels_in * 9;
                float* out_o = out_bt + o * 16;

                // One accumulator per destination blade. Blades with two inflow paths get a second accumulator so each chain stays length 1 over the i loop.
                float a0 = 0.f;
                float a1_diag = 0.f, a1_off = 0.f;
                float a2 = 0.f, a3 = 0.f, a4 = 0.f;
                float a5_diag = 0.f, a5_off = 0.f;
                float a6_diag = 0.f, a6_off = 0.f;
                float a7_diag = 0.f, a7_off = 0.f;
                float a8 = 0.f, a9 = 0.f, a10 = 0.f;
                float a11_diag = 0.f, a11_off = 0.f;
                float a12_diag = 0.f, a12_off = 0.f;
                float a13_diag = 0.f, a13_off = 0.f;
                float a14 = 0.f;
                float a15_diag = 0.f, a15_off = 0.f;

                for (int i = 0; i < channels_in; i++) {
                    const float* wi = w_o + i * 9;
                    const float* xi = x_bt + i * 16;

                    // Pre-scale the 9 weights by the basis norm so the body
                    // below is just FMAs.
                    const float w0 = wi[0] * C1;
                    const float w1 = wi[1] * C2;
                    const float w2 = wi[2] * C6;
                    const float w3 = wi[3] * C2;
                    const float w4 = wi[4] * C1;
                    const float w5 = wi[5] * C1;
                    const float w6 = wi[6] * C3;
                    const float w7 = wi[7] * C3;
                    const float w8 = wi[8] * C1;

                    // diagonal slices (w = 0..4): d == s
                    a0       += w0 * xi[0];
                    a1_diag  += w1 * xi[1];
                    a2       += w1 * xi[2];
                    a3       += w1 * xi[3];
                    a4       += w1 * xi[4];
                    a5_diag  += w2 * xi[5];
                    a6_diag  += w2 * xi[6];
                    a7_diag  += w2 * xi[7];
                    a8       += w2 * xi[8];
                    a9       += w2 * xi[9];
                    a10      += w2 * xi[10];
                    a11_diag += w3 * xi[11];
                    a12_diag += w3 * xi[12];
                    a13_diag += w3 * xi[13];
                    a14      += w3 * xi[14];
                    a15_diag += w4 * xi[15];

                    // grade-lowering slices (w = 5..8): d != s
                    a1_off  += w5 * xi[0];   // (1, 0)
                    a5_off  += w6 * xi[2];   // (5, 2)
                    a6_off  += w6 * xi[3];   // (6, 3)
                    a7_off  += w6 * xi[4];   // (7, 4)
                    a11_off += w7 * xi[8];   // (11, 8)
                    a12_off += w7 * xi[9];   // (12, 9)
                    a13_off += w7 * xi[10];  // (13, 10)
                    a15_off += w8 * xi[14];  // (15, 14)
                }

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
 
// wrapper
torch::Tensor asl_equi_linear(torch::Tensor x, torch::Tensor weight, torch::Tensor basis,
    std::optional<torch::Tensor> bias = std::nullopt) {
    CYCLE_SCOPE("equi_linear");
    x = x.contiguous();
    weight = weight.contiguous();
    basis = basis.contiguous();
    
    // sizing our output based on arguments
    int B = x.size(0);
    int T = x.size(1);
    int channels_in = x.size(2);
    int channels_out = weight.size(0);
    auto out = torch::zeros({B, T, channels_out, 16}, x.options());
    
    //extract bias pointer
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().contiguous().data_ptr<float>();
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

// Python bindings
#include <pybind11/pybind11.h>

PYBIND11_MODULE(linear, m) {
    m.def("geometric_product", &geometric_product, "Geometric product");
    m.def("equi_linear", &asl_equi_linear, "pin-equivariant linear map",
        pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("basis"), pybind11::arg("bias") = std::nullopt);
    BIND_CYCLE_COUNTER(m);
}
