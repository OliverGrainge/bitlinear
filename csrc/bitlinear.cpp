#include <torch/extension.h>

// Forward declarations for CPU implementation
torch::Tensor bitlinear_cpu(
    torch::Tensor x,
    torch::Tensor w_scale,
    torch::Tensor w_packed,
    c10::optional<torch::Tensor> bias_opt,
    float eps
);

std::tuple<torch::Tensor, torch::Tensor> prepare_weights_cpu(
    torch::Tensor weight,
    float eps,
    std::string quant_type
);

// Forward declarations for CUDA implementation
#ifdef WITH_CUDA
#include "cuda/bitlinear_cuda.h"
#endif

// ============================================================================
// DISPATCHER FUNCTIONS
// ============================================================================

torch::Tensor bitlinear(
    torch::Tensor x,
    torch::Tensor w_scale,
    torch::Tensor w_packed,
    c10::optional<torch::Tensor> bias_opt,
    float eps
) {
    // Check device and dispatch
    if (x.is_cuda()) {
#ifdef WITH_CUDA
        return bitlinear_cuda(x, w_scale, w_packed, bias_opt, eps);
#else
        AT_ERROR("bitlinear: CUDA kernel requested but not compiled with CUDA support");
#endif
    } else {
        return bitlinear_cpu(x, w_scale, w_packed, bias_opt, eps);
    }
}

std::tuple<torch::Tensor, torch::Tensor> prepare_weights(
    torch::Tensor weight,
    float eps,
    std::string quant_type
) {
    // Check device and dispatch
    if (weight.is_cuda()) {
#ifdef WITH_CUDA
        return prepare_weights_cuda(weight, eps, quant_type);
#else
        AT_ERROR("prepare_weights: CUDA kernel requested but not compiled with CUDA support");
#endif
    } else {
        return prepare_weights_cpu(weight, eps, quant_type);
    }
}

// ============================================================================
// PYTHON BINDINGS
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bitlinear", &bitlinear, "BitLinear forward (auto-dispatch)",
          py::arg("x"),
          py::arg("w_scale"),
          py::arg("w_packed"),
          py::arg("bias") = c10::nullopt,
          py::arg("eps") = 1e-5f);
    
    m.def("prepare_weights", &prepare_weights, "Prepare weights",
          py::arg("weight"),
          py::arg("eps") = 1e-5f,
          py::arg("quant_type") = "per_tensor");
}