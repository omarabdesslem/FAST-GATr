#include <torch/extension.h>
#include <map>
#include <tuple>
#include <torch/script.h>
#include <cstring>
#include <vector>
#include "_cycle_counter.hpp"

// replacement for _BASIS_CACHE
//using device type&index to identify device
using devScalTuple = std::tuple<torch::DeviceType, torch::DeviceIndex, torch::ScalarType>;
static std::map<devScalTuple, torch::Tensor> basis_cache_gp, basis_cache_op;

//defines bilinear types geometric or outer prod.
enum class Kind { GP, OP };


// C cache for basis using float* arrays
using basisPtrTuple = std::tuple<torch::DeviceType, torch::DeviceIndex>;
static std::map<basisPtrTuple, float*> basis_cache_gp_raw, basis_cache_op_raw;

float* _load_bilinear_basis_raw(
    Kind kind) {
    /*
    Load the bilinear basis as raw float array (shape: 16*16*16 = 4096).
    Load from .pt file on first call, then cache in memory.
    */
    
    auto& cache = (kind == Kind::GP) ? basis_cache_gp_raw : basis_cache_op_raw;
    auto key = std::make_tuple(torch::kCPU, static_cast<torch::DeviceIndex>(0));
    
    // Check if already cached
    auto found = cache.find(key);
    if (found != cache.end()) {
        return found->second;
    }
    
    // Load from file if not cached
    std::string filename = (kind == Kind::GP)
        ? "basis/geometric_product_cpp.pt"
        : "basis/outer_product_cpp.pt";
    
    auto module = torch::jit::load(filename);
    torch::Tensor tensor_basis = module.attr("b").toTensor();
    tensor_basis = tensor_basis.contiguous().to(torch::kFloat32);
    
    // Allocate raw float array and copy data
    int64_t size = tensor_basis.numel();  // Should be 4096 (16*16*16)
    float* raw_basis = new float[size];
    std::memcpy(raw_basis, tensor_basis.data_ptr<float>(), size * sizeof(float));
    
    // Cache it
    cache[key] = raw_basis;
    return raw_basis;
}

torch::Tensor _load_bilinear_basis(
    Kind kind,
    torch::Device device,
    torch::ScalarType dtype) {
    /*
    Load the bilinear basis for geometric product or outer product.

    The bilinear basis is a 3D tensor with shape (16, 16, 16). One can
    understand the basis tensor by thinking of how the calculation of
    the bilinear maps between two multi-vectors expressed as coefficient
    vectors w.r.t. k-blades. When two multi-vectors are multiplied, each
    new coefficient corresponding to the k-blade comes from the "cartesian
    product" of the coefficients of the two multi-vectors. For example,
    the coefficient of the 1-blade ``e_1`` comes from multiple sources
    including ``gp(e_1, 1)`` and ``gp(e_0, e_01)``. So, the basis tensor
    actually defines a "computation graph" for the bilinear maps.

    The source basis are stored under the ``basis`` directory in `torch.float32`.
    The basis loader use the ``torch.float32`` tensor loaded to CPU as the prototype
    for all other devices and data types.

    Parameters
    ----------
    kind : Literal["gp", "op"]
        Kind of the bilinear basis, ``"gp"`` for geometric product and
        ``"op"`` for outer (wedge) product.
    device : torch.device
        Device for the basis.
    dtype : torch.dtype
        Data type for the basis.

    Returns
    -------
    torch.Tensor
        Bilinear basis with shape (16, 16, 16).
    */
    
    auto& cache = (kind == Kind::GP) ? basis_cache_gp : basis_cache_op;
    auto key = std::make_tuple(device.type(), device.index(), dtype);
    
    // Check cache for exact device/dtype
    auto basis = cache.find(key);
    if (basis != cache.end()) {
        return basis->second;
    }
    
    // Load raw basis data
    float* raw_data = _load_bilinear_basis_raw(kind);
    
    // Create torch tensor from raw data
    torch::Tensor res = torch::from_blob(
        raw_data, 
        {16, 16, 16},
        torch::dtype(torch::kFloat32)
    ).clone().to(device, dtype);
    
    cache[key] = res;
    return res;
}

torch::Tensor _compute_inner_product_selector(
    torch::Device device,
    bool keep_tri_vector = true
){ 
    /*
    Load the indices for PGA inner product to the device.

    PGA inner product operation exclude the coefficients corresponding to
    basis containing ``e_0``. The indices are hard-coded here. The reason
    to have this cached function is to avoid repeated copying from CPU to
    target multi-vector device using the ``torch.Tensor.to`` method.

    Parameters
    ----------
    device : torch.device
        Device for the indices.
    keep_tri_vector : bool
        Drop the tri-vector index (``e_123``) if set to ``False``. This is
        used in the inner-product component of the geometric attention.

    Returns
    -------
        torch.Tensor
    */
    std::vector<int64_t> idx = {0, 2, 3, 4, 8, 9, 10, 14};
    if (!keep_tri_vector) {
        idx.pop_back();
    }
    
    // Build on CPU first, then move to target device.
    torch::Tensor result = torch::empty(idx.size(), torch::dtype(torch::kInt64).device(torch::kCPU));
    int64_t* result_ptr = result.data_ptr<int64_t>();
    
    for (size_t i = 0; i < idx.size(); ++i) {
        result_ptr[i] = idx[i];
    }
    
    return result.to(device);
}

// geometric_product(x, y) -> res
// torch tensors x, y, res: shape (..., 16),
torch::Tensor geometric_product(
    torch::Tensor x, 
    torch::Tensor y) {
    /*Geometric product between two batches of multi-vectors.

    The input tensors ``x`` and ``y`` are multi-vectors with shape (..., 16).
    where ``...`` dimensions can denote batches or batches plus channels.
    When channel dimensions are present, the geometric product is calculated
    channel-wise (and batch-wise). For instance, the first channel of ``x[0]``
    is multiplied with the first channel of ``y[0]``, and so on. No channel-mixing
    here.*/
    
    TORCH_CHECK(x.sizes() == y.sizes(), "geometric_product: x and y must have identical shapes");
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) == 16, "geometric_product: last dimension must be 16");

    CYCLE_SCOPE("geometric_product");

    const auto out_device = x.device();
    const auto out_dtype = x.scalar_type();

    auto x_cpu = x.contiguous().to(torch::kCPU, torch::kFloat32);
    auto y_cpu = y.contiguous().to(torch::kCPU, torch::kFloat32);
    const float* x_ptr = x_cpu.data_ptr<float>();
    const float* y_ptr = y_cpu.data_ptr<float>();

    // Load basis ptr
    float* basis_ptr = _load_bilinear_basis_raw(Kind::GP);
    
    // Allocate result array
    int64_t batch_size = x_cpu.numel() / 16;
    auto result_cpu = torch::zeros_like(x_cpu);
    float* result_ptr = result_cpu.data_ptr<float>();
    
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int i = 0; i < 16; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < 16; ++j) {
                for (int k = 0; k < 16; ++k) {
                    int basis_idx = i * 256 + j * 16 + k;  // 16*16 = 256
                    float basis_val = basis_ptr[basis_idx];
                    float x_val = x_ptr[b * 16 + j];
                    float y_val = y_ptr[b * 16 + k];
                    sum += basis_val * x_val * y_val;
                }
            }
            result_ptr[b * 16 + i] = sum;
        }
    }
    
    return result_cpu.to(out_device, out_dtype);
}

// outer_product(x, y) -> res
// torch tensors x, y, res: shape (..., 16),
torch::Tensor outer_product(
    torch::Tensor x, 
    torch::Tensor y) {
    /*Outer product between two batches of multi-vectors.

    The input tensors ``x`` and ``y`` are multi-vectors with shape (..., 16).
    where ``...`` dimensions can denote batches or batches plus channels. When
    channel dimensions are present, the outer product is calculated channel-wise
    (and batch-wise). For instance, the first channel of ``x[0]`` is multiplied
    with the first channel of ``y[0]``, and so on. No channel-mixing here.
    */
    
    TORCH_CHECK(x.sizes() == y.sizes(), "outer_product: x and y must have identical shapes");
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) == 16, "outer_product: last dimension must be 16");

    CYCLE_SCOPE("outer_product");

    const auto out_device = x.device();
    const auto out_dtype = x.scalar_type();

    auto x_cpu = x.contiguous().to(torch::kCPU, torch::kFloat32);
    auto y_cpu = y.contiguous().to(torch::kCPU, torch::kFloat32);
    const float* x_ptr = x_cpu.data_ptr<float>();
    const float* y_ptr = y_cpu.data_ptr<float>();

    // Load basis
    float* basis_ptr = _load_bilinear_basis_raw(Kind::OP);
    
    // Allocate result array
    int64_t batch_size = x_cpu.numel() / 16;
    auto result_cpu = torch::zeros_like(x_cpu);
    float* result_ptr = result_cpu.data_ptr<float>();
    
    // compute outer product
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int i = 0; i < 16; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < 16; ++j) {
                for (int k = 0; k < 16; ++k) {
                    int basis_idx = i * 256 + j * 16 + k;  // 16*16 = 256
                    float basis_val = basis_ptr[basis_idx];
                    float x_val = x_ptr[b * 16 + j];
                    float y_val = y_ptr[b * 16 + k];
                    sum += basis_val * x_val * y_val;
                }
            }
            result_ptr[b * 16 + i] = sum;
        }
    }
    
    return result_cpu.to(out_device, out_dtype);
}

// Equilinear function substitute
//   x:      [batch][tokens][channnels_in][16]
//   weight: [out_ch][channels_in][9]
//   basis:  [9][16][16]
//   out:    [batch][tokens][channels_out][16]
//   bias:   [out_ch]
 
void equi_linear_kernel(float* out, const float* x, const float* weight, const float* basis, const float* bias,
    int B, int T, int channels_in, int channels_out) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int batch_token = b * T + t;
            for (int o = 0; o < channels_out; o++) {
                for (int d = 0; d < 16; d++) {
                    float sum = 0.0f;

                    for (int i = 0; i < channels_in; i++) {
                        for (int w = 0; w < 9; w++) {
                            for (int s = 0; s < 16; s++) {
                                float wt = weight[o * channels_in * 9 + i * 9 + w];
                                float bs = basis[w * 16 * 16 + d * 16 + s];
                                float xv = x[batch_token * channels_in * 16 + i * 16 + s];

                                sum += wt * bs * xv;
                            }
                        }
                    }

                    out[batch_token * channels_out * 16 + o * 16 + d] = sum;
                }

                if (bias != NULL) {
                    out[batch_token * channels_out * 16 + o * 16 + 0] += bias[o];
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
    m.def("outer_product", &outer_product, "Outer product");
    m.def("_compute_inner_product_selector", &_compute_inner_product_selector, "_compute_inner_product_selector");
    m.def("equi_linear", &asl_equi_linear, "pin-equivariant linear map",
        pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("basis"), pybind11::arg("bias") = std::nullopt);
    BIND_CYCLE_COUNTER(m);
}