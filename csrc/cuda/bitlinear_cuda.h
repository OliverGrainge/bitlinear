// csrc/cuda/bitlinear_cuda.h
#pragma once
#include <torch/extension.h>

/**
 * Performs BitLinear forward pass on CUDA.
 * 
 * This function automatically selects the most efficient kernel based on
 * the problem size:
 * - Small batch (M <= 32): Warp-per-row kernel
 * - Large problems (M,N,K >= 128): Tiled kernel with shared memory
 * - Medium problems: Vectorized kernel
 * 
 * @param x Input activations [M, K]
 * @param w_scale Weight scale (scalar tensor)
 * @param w_packed Packed weights [N, K/4] (2-bit packed)
 * @param bias_opt Optional bias [N]
 * @param eps Epsilon for numerical stability
 * @return Output tensor [M, N] in same dtype as input
 */
torch::Tensor bitlinear_cuda(
    torch::Tensor x,
    torch::Tensor w_scale,
    torch::Tensor w_packed,
    c10::optional<torch::Tensor> bias_opt,
    float eps
);

/**
 * Prepares and packs weights for BitLinear on CUDA.
 * 
 * Quantizes weights to {-1, 0, 1} and packs them into 2-bit format
 * (4 weights per byte).
 * 
 * @param weight Input weights [N, K]
 * @param eps Epsilon for scale computation
 * @param quant_type Quantization type (currently unused, for future extension)
 * @return Tuple of (weight_scale [scalar], packed_weights [N, K/4])
 */
std::tuple<torch::Tensor, torch::Tensor> prepare_weights_cuda(
    torch::Tensor weight,
    float eps,
    std::string quant_type
);
