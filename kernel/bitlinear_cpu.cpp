#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <omp.h>
#include <cstring>
#include <arm_neon.h>

// ============================================================================
// SHARED UTILITIES
// ============================================================================

#include "arm/ukernel.cpp"
// ============================================================================
// BATCH SIZE = 1 OPTIMIZED KERNEL
// ============================================================================

// For batch_size=1, we don't need M-tiling and can simplify the kernel.
// We process the single input row with better cache locality and less overhead.
constexpr int64_t TILE_N_BS1 = 128;  // Larger N tile since we only have 1 row
constexpr int64_t TILE_K_BS1 = 512;  // Larger K tile for better throughput

torch::Tensor bitlinear_bs1(
    torch::Tensor x,
    torch::Tensor w_scale,
    torch::Tensor w_packed,
    c10::optional<torch::Tensor> bias_opt,
    float eps
) {
    // Activation quantization
    auto x_fp32 = x.to(torch::kFloat32);
    auto x_scale = std::get<0>(x_fp32.abs().max(-1, true)).clamp_min(eps) / 127.0;
    auto x_quant = (x_fp32 / x_scale).round().clamp(-128, 127).to(torch::kInt8);
    x_scale = x_scale.squeeze(-1).contiguous();
    
    // Get data pointers
    auto x_quant_ptr  = x_quant.data_ptr<int8_t>();
    auto w_scale_ptr  = w_scale.data_ptr<float>();
    auto x_scale_ptr  = x_scale.data_ptr<float>();
    auto w_packed_ptr = w_packed.data_ptr<uint8_t>();
    
    float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias = bias_opt->to(torch::kFloat32);
        bias_ptr  = bias.data_ptr<float>();
    }

    // Dimensions
    const int64_t K = x.size(1);
    const int64_t N = w_packed.size(0);
    TORCH_CHECK(K % 4 == 0, "K must be a multiple of 4");

    // Output buffer
    auto y = torch::zeros({1, N}, torch::kFloat32);
    auto y_ptr = y.data_ptr<float>();

    const float w_scale_val = w_scale_ptr[0];
    const float a_scale = x_scale_ptr[0];
    const float combined_scale = a_scale * w_scale_val;

    // Process in parallel over N dimension (output features)
    #pragma omp parallel
    {
        alignas(64) int8_t w_buffer[TILE_N_BS1 * TILE_K_BS1];
        alignas(64) int32_t acc_buffer[TILE_N_BS1];

        #pragma omp for schedule(dynamic)
        for (int64_t n_tile = 0; n_tile < N; n_tile += TILE_N_BS1) {
            const int64_t n_end = std::min(n_tile + TILE_N_BS1, N);
            const int64_t n_size = n_end - n_tile;

            // Zero accumulators
            std::memset(acc_buffer, 0, n_size * sizeof(int32_t));

            // K tiling
            for (int64_t k_tile = 0; k_tile < K; k_tile += TILE_K_BS1) {
                const int64_t k_end = std::min(k_tile + TILE_K_BS1, K);
                const int64_t k_size = k_end - k_tile;

                // Unpack weights into cache
                for (int64_t n = 0; n < n_size; ++n) {
                    const int64_t global_n = n_tile + n;

                    for (int64_t k = 0; k < k_size; k += 4) {
                        const int64_t global_k = k_tile + k;
                        uint8_t w_byte = w_packed_ptr[global_n * (K / 4) + (global_k / 4)];

                        w_buffer[n * k_size + k + 0] = static_cast<int8_t>((w_byte >> 6) & 0x03) - 1;
                        w_buffer[n * k_size + k + 1] = static_cast<int8_t>((w_byte >> 4) & 0x03) - 1;
                        w_buffer[n * k_size + k + 2] = static_cast<int8_t>((w_byte >> 2) & 0x03) - 1;
                        w_buffer[n * k_size + k + 3] = static_cast<int8_t>(w_byte & 0x03) - 1;
                    }
                }

                // Compute dot products - use 1x4 kernel
                const int8_t* x_row = x_quant_ptr + k_tile;

                int64_t n = 0;
                for (; n + 3 < n_size; n += 4) {
                    const int8_t* b0 = w_buffer + (n + 0) * k_size;
                    const int8_t* b1 = w_buffer + (n + 1) * k_size;
                    const int8_t* b2 = w_buffer + (n + 2) * k_size;
                    const int8_t* b3 = w_buffer + (n + 3) * k_size;

                    i8dot_1x4(
                        x_row, b0, b1, b2, b3,
                        acc_buffer[n + 0],
                        acc_buffer[n + 1],
                        acc_buffer[n + 2],
                        acc_buffer[n + 3],
                        static_cast<int32_t>(k_size)
                    );
                }

                // Handle remaining outputs
                for (; n < n_size; ++n) {
                    acc_buffer[n] += i8dot(
                        x_row,
                        w_buffer + n * k_size,
                        static_cast<int32_t>(k_size)
                    );
                }
            }

            // Scale and write results
            for (int64_t n = 0; n < n_size; ++n) {
                const int64_t global_n = n_tile + n;
                float result = static_cast<float>(acc_buffer[n]) * combined_scale;

                if (bias_ptr) {
                    result += bias_ptr[global_n];
                }

                y_ptr[global_n] = result;
            }
        }
    }

    return y.to(x.dtype());
}

// ============================================================================
// BATCH SIZE > 1 OPTIMIZED KERNEL (ORIGINAL TILED IMPLEMENTATION)
// ============================================================================

constexpr int64_t TILE_M = 32;
constexpr int64_t TILE_N = 64;
constexpr int64_t TILE_K = 256;

torch::Tensor bitlinear_batched(
    torch::Tensor x,
    torch::Tensor w_scale,
    torch::Tensor w_packed,
    c10::optional<torch::Tensor> bias_opt,
    float eps
) {
    // Activation quantization
    auto x_fp32 = x.to(torch::kFloat32);
    auto x_scale = std::get<0>(x_fp32.abs().max(-1, true)).clamp_min(eps) / 127.0;
    auto x_quant = (x_fp32 / x_scale).round().clamp(-128, 127).to(torch::kInt8);
    x_scale = x_scale.squeeze(-1).contiguous();
    
    // Get data pointers
    auto x_quant_ptr  = x_quant.data_ptr<int8_t>();
    auto w_scale_ptr  = w_scale.data_ptr<float>();
    auto x_scale_ptr  = x_scale.data_ptr<float>();
    auto w_packed_ptr = w_packed.data_ptr<uint8_t>();
    
    float* bias_ptr = nullptr;
    if (bias_opt.has_value()) {
        auto bias = bias_opt->to(torch::kFloat32);
        bias_ptr  = bias.data_ptr<float>();
    }

    // Dimensions
    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = w_packed.size(0);
    TORCH_CHECK(K % 4 == 0, "K must be a multiple of 4");

    // Output buffer
    auto y = torch::zeros({M, N}, torch::kFloat32);
    auto y_ptr = y.data_ptr<float>();

    const float w_scale_val = w_scale_ptr[0];

    // Tiled matmul
    #pragma omp parallel
    {
        alignas(64) int8_t  x_buffer[TILE_M * TILE_K];
        alignas(64) int8_t  w_buffer[TILE_N * TILE_K];
        alignas(64) int32_t acc_buffer[TILE_M * TILE_N];

        #pragma omp for collapse(2) schedule(dynamic)
        for (int64_t m_tile = 0; m_tile < M; m_tile += TILE_M) {
            for (int64_t n_tile = 0; n_tile < N; n_tile += TILE_N) {
                const int64_t m_end  = std::min(m_tile + TILE_M, M);
                const int64_t n_end  = std::min(n_tile + TILE_N, N);
                const int64_t m_size = m_end - m_tile;
                const int64_t n_size = n_end - n_tile;

                std::memset(acc_buffer, 0, m_size * n_size * sizeof(int32_t));

                // K tiling
                for (int64_t k_tile = 0; k_tile < K; k_tile += TILE_K) {
                    const int64_t k_end  = std::min(k_tile + TILE_K, K);
                    const int64_t k_size = k_end - k_tile;

                    // Load activations
                    for (int64_t m = 0; m < m_size; ++m) {
                        const int64_t global_m = m_tile + m;
                        std::memcpy(
                            &x_buffer[m * k_size],
                            &x_quant_ptr[global_m * K + k_tile],
                            k_size * sizeof(int8_t)
                        );
                    }

                    // Unpack weights
                    for (int64_t n = 0; n < n_size; ++n) {
                        const int64_t global_n = n_tile + n;

                        for (int64_t k = 0; k < k_size; k += 4) {
                            const int64_t global_k = k_tile + k;
                            uint8_t w_byte = w_packed_ptr[global_n * (K / 4) + (global_k / 4)];

                            w_buffer[n * k_size + k + 0] = static_cast<int8_t>((w_byte >> 6) & 0x03) - 1;
                            w_buffer[n * k_size + k + 1] = static_cast<int8_t>((w_byte >> 4) & 0x03) - 1;
                            w_buffer[n * k_size + k + 2] = static_cast<int8_t>((w_byte >> 2) & 0x03) - 1;
                            w_buffer[n * k_size + k + 3] = static_cast<int8_t>(w_byte & 0x03) - 1;
                        }
                    }

                    // Compute tile
                    for (int64_t m = 0; m < m_size; ++m) {
                        const int8_t* x_row = x_buffer + m * k_size;

                        int64_t n = 0;
                        for (; n + 3 < n_size; n += 4) {
                            const int8_t* b0 = w_buffer + (n + 0) * k_size;
                            const int8_t* b1 = w_buffer + (n + 1) * k_size;
                            const int8_t* b2 = w_buffer + (n + 2) * k_size;
                            const int8_t* b3 = w_buffer + (n + 3) * k_size;

                            int32_t& c0 = acc_buffer[m * n_size + (n + 0)];
                            int32_t& c1 = acc_buffer[m * n_size + (n + 1)];
                            int32_t& c2 = acc_buffer[m * n_size + (n + 2)];
                            int32_t& c3 = acc_buffer[m * n_size + (n + 3)];

                            i8dot_1x4(
                                x_row, b0, b1, b2, b3,
                                c0, c1, c2, c3,
                                static_cast<int32_t>(k_size)
                            );
                        }

                        for (; n < n_size; ++n) {
                            int32_t& c = acc_buffer[m * n_size + n];
                            c += i8dot(
                                x_row,
                                w_buffer + n * k_size,
                                static_cast<int32_t>(k_size)
                            );
                        }
                    }
                }

                // Scale and write results
                for (int64_t m = 0; m < m_size; ++m) {
                    const int64_t global_m = m_tile + m;
                    const float   a_scale  = x_scale_ptr[global_m];
                    const float   combined_scale = a_scale * w_scale_val;

                    for (int64_t n = 0; n < n_size; ++n) {
                        const int64_t global_n = n_tile + n;

                        float result = static_cast<float>(acc_buffer[m * n_size + n]) * combined_scale;

                        if (bias_ptr) {
                            result += bias_ptr[global_n];
                        }

                        y_ptr[global_m * N + global_n] = result;
                    }
                }
            }
        }
    }

    return y.to(x.dtype());
}

// ============================================================================
// UNIFIED INTERFACE
// ============================================================================

torch::Tensor bitlinear(
    torch::Tensor x,
    torch::Tensor w_scale,
    torch::Tensor w_packed,
    c10::optional<torch::Tensor> bias_opt,
    float eps
) {
    const int64_t batch_size = x.size(0);
    
    if (batch_size == 1) {
        return bitlinear_bs1(x, w_scale, w_packed, bias_opt, eps);
    } else {
        return bitlinear_batched(x, w_scale, w_packed, bias_opt, eps);
    }
}

// ============================================================================
// WEIGHT PREPARATION (SHARED)
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor> prepare_weights(
    torch::Tensor weight,
    float eps,
    std::string quant_type
) {
    auto w_fp32 = weight.to(torch::kFloat32);
    auto w_scale = w_fp32.abs().mean().clamp_min(eps);
    auto w_quant = ((w_fp32 / w_scale).round().clamp(-1, 1) + 1).to(torch::kInt8);

    const int64_t N = weight.size(0);
    const int64_t K = weight.size(1);
    TORCH_CHECK(K % 4 == 0, "K must be a multiple of 4");

    auto w_quant_ptr = w_quant.data_ptr<int8_t>();
    auto w_packed = torch::zeros({N, K / 4}, torch::kUInt8);
    auto w_packed_ptr = w_packed.data_ptr<uint8_t>();

    #pragma omp parallel for
    for (int64_t n = 0; n < N; n++) {
        for (int64_t k = 0; k < K; k += 4) {
            uint8_t w0 = static_cast<uint8_t>(w_quant_ptr[n * K + k]);
            uint8_t w1 = static_cast<uint8_t>(w_quant_ptr[n * K + k + 1]);
            uint8_t w2 = static_cast<uint8_t>(w_quant_ptr[n * K + k + 2]);
            uint8_t w3 = static_cast<uint8_t>(w_quant_ptr[n * K + k + 3]);

            w_packed_ptr[n * (K / 4) + (k / 4)] = (w0 << 6) | (w1 << 4) | (w2 << 2) | w3;
        }
    }
    
    return std::make_tuple(w_scale.to(weight.dtype()), w_packed);
}

// ============================================================================
// PYTHON BINDINGS
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bitlinear", &bitlinear, "BitLinear forward (auto-dispatch)");
    m.def("prepare_weights", &prepare_weights, "Prepare weights");
}