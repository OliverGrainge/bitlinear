#pragma once
#include <torch/extension.h>

torch::Tensor bitlinear_cuda(
    torch::Tensor x,
    torch::Tensor w_scale,
    torch::Tensor w_packed,
    c10::optional<torch::Tensor> bias_opt,
    float eps
);

std::tuple<torch::Tensor, torch::Tensor> prepare_weights_cuda(
    torch::Tensor weight,
    float eps,
    std::string quant_type
);