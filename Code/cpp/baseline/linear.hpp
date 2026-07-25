#pragma once
#include <torch/torch.h>
#include <optional>

torch::Tensor outer_product(torch::Tensor x, torch::Tensor y);
torch::Tensor geometric_product(torch::Tensor x, torch::Tensor y);
torch::Tensor equi_dual(const torch::Tensor& x);
torch::Tensor equi_join(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const std::optional<torch::Tensor>& reference = std::nullopt
);
torch::Tensor _compute_inner_product_selector(
    torch::Device device,
    bool keep_tri_vector = true
);
torch::Tensor asl_equi_linear(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor basis,
    c10::optional<torch::Tensor> bias = c10::nullopt
);