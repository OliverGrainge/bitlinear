// csrc/cuda/bitlinear_cuda.cu
#include <torch/extension.h>

__global__ void bitlinear_cuda_kernel(
    const float* __restrict__ x_scale_ptr,
    const int8_t* __restrict__ x_quant_ptr, 
    const float* __restrict__ w_scale_ptr, 
    const uint8_t* __restrict__ w_packed_ptr,  
    const float* __restrict__ bias_ptr, 
    float* __restrict__ y_ptr,
    int M, int K, int N) {

    int row = blockIdx.x * blockDim.x + threadIdx.x; 
    int col = blockIdx.y * blockDim.y + threadIdx.y; 

    if (row < M && col < N) {
        // Accumulate dot product in integer domain (as float accumulator)
        float acc = 0.0f;

        for (int k = 0; k < K; k += 4) {
            int8_t x0 = x_quant_ptr[row * K + k + 0];
            int8_t x1 = x_quant_ptr[row * K + k + 1];
            int8_t x2 = x_quant_ptr[row * K + k + 2];
            int8_t x3 = x_quant_ptr[row * K + k + 3];

            // weights packed as 2 bits each, values in {0,1,2} encoded then shifted to {-1,0,1}
            uint8_t w_byte = w_packed_ptr[col * (K / 4) + (k / 4)];
            int8_t w0 = static_cast<int8_t>((w_byte >> 6) & 0x03) - 1;
            int8_t w1 = static_cast<int8_t>((w_byte >> 4) & 0x03) - 1;
            int8_t w2 = static_cast<int8_t>((w_byte >> 2) & 0x03) - 1;
            int8_t w3 = static_cast<int8_t>( w_byte        & 0x03) - 1;

            acc += x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
        }

        // x ≈ x_scale[row] * x_quant, w ≈ w_scale * w_quant
        // y ≈ bias + (Σ x_q * w_q) * x_scale[row] * w_scale
        float scaled = acc * x_scale_ptr[row] * w_scale_ptr[0];
        float bias   = bias_ptr[col];
        y_ptr[row * N + col] = scaled + bias;
    }
}

// x: [M, K], w_scale: scalar (from prepare_weights), w_packed: [N, K/4]
torch::Tensor bitlinear_cuda(torch::Tensor x,
                             torch::Tensor w_scale,
                             torch::Tensor w_packed,
                             c10::optional<torch::Tensor> bias_opt,
                             float eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w_scale.is_cuda(), "w_scale must be a CUDA tensor");
    TORCH_CHECK(w_packed.is_cuda(), "w_packed must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K]");
    TORCH_CHECK(w_packed.dim() == 2, "w_packed must be 2D [N, K/4]");

    auto M = static_cast<int>(x.size(0));
    auto K = static_cast<int>(x.size(1));
    auto N = static_cast<int>(w_packed.size(0));

    TORCH_CHECK(K % 4 == 0, "K must be divisible by 4");
    TORCH_CHECK(w_packed.size(1) == K / 4,
                "w_packed has inconsistent second dimension (expected K/4)");

    // ---- Activation quantization ----
    // x_scale: [M, 1] then squeezed to [M]
    auto x_fp32 = x.to(torch::kFloat32);
    auto x_scale = std::get<0>(x_fp32.abs().max(-1, true)).clamp_min(eps) / 127.0;
    auto x_quant = (x_fp32 / x_scale)
                       .round()
                       .clamp(-128, 127)
                       .to(torch::kInt8)
                       .contiguous();
    x_scale = x_scale.squeeze(-1).contiguous();  // [M]

    // ---- Data pointers ----
    const int8_t*  x_quant_ptr  = x_quant.data_ptr<int8_t>();
    const float*   w_scale_ptr  = w_scale.to(torch::kFloat32).contiguous().data_ptr<float>();
    const float*   x_scale_ptr  = x_scale.data_ptr<float>();
    const uint8_t* w_packed_ptr = w_packed.data_ptr<uint8_t>();

    auto opts_f32 = x.options().dtype(torch::kFloat32);
    auto y = torch::zeros({M, N}, opts_f32).contiguous();
    float* y_ptr = y.data_ptr<float>();

    // ---- Bias handling (optional) ----
    torch::Tensor bias;
    if (bias_opt.has_value()) {
        bias = bias_opt->to(torch::kFloat32).contiguous();
        TORCH_CHECK(bias.numel() == N,
                    "bias must have size N (got ", bias.numel(), ", expected ", N, ")");
    } else {
        bias = torch::zeros({N}, opts_f32).contiguous();
    }
    const float* bias_ptr = bias.data_ptr<float>();

    // ---- Launch kernel ----
    dim3 block(32, 32);
    dim3 grid((M + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    bitlinear_cuda_kernel<<<grid, block>>>(
        x_scale_ptr,
        x_quant_ptr,
        w_scale_ptr,
        w_packed_ptr,
        bias_ptr,
        y_ptr,
        M, K, N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "bitlinear_cuda_kernel launch failed: ",
                cudaGetErrorString(err));

    return y;
}

// ----------------- weight packing -----------------

__global__ void prepare_weights_cuda_kernel(const int8_t* __restrict__ w_quant_ptr,
                                            uint8_t* __restrict__ w_packed_ptr,
                                            int N,
                                            int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        for (int k = 0; k < K; k += 4) {
            uint8_t w0 = static_cast<uint8_t>(w_quant_ptr[row * K + k + 0]);
            uint8_t w1 = static_cast<uint8_t>(w_quant_ptr[row * K + k + 1]);
            uint8_t w2 = static_cast<uint8_t>(w_quant_ptr[row * K + k + 2]);
            uint8_t w3 = static_cast<uint8_t>(w_quant_ptr[row * K + k + 3]);

            w_packed_ptr[row * (K / 4) + (k / 4)] =
                static_cast<uint8_t>((w0 << 6) | (w1 << 4) | (w2 << 2) | w3);
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> prepare_weights_cuda(
    torch::Tensor weight,
    float eps,
    std::string /*quant_type*/) {

    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    const int64_t N = weight.size(0);
    const int64_t K = weight.size(1);

    auto opts_f32 = weight.options().dtype(torch::kFloat32);
    auto opts_u8  = weight.options().dtype(torch::kUInt8);
    auto opts_i8  = weight.options().dtype(torch::kInt8);

    auto w_fp32 = weight.to(opts_f32);     // [N, K] float32

    // Single global scale for weights
    auto w_scale = w_fp32.abs().mean().clamp_min(eps);

    // Map to {-1, 0, 1} via {-1,0,1} + 1 -> {0,1,2} for packing
    auto w_quant = ((w_fp32 / w_scale)
                        .round()
                        .clamp(-1, 1) + 1)
                       .to(opts_i8)
                       .contiguous();

    TORCH_CHECK(K % 4 == 0, "K must be divisible by 4 for this packing");

    auto w_packed = torch::zeros({N, K / 4}, opts_u8).contiguous();

    const int8_t*  w_quant_ptr  = w_quant.data_ptr<int8_t>();
    uint8_t*       w_packed_ptr = w_packed.data_ptr<uint8_t>();

    dim3 block(32 * 32);
    dim3 grid((N + block.x - 1) / block.x);

    prepare_weights_cuda_kernel<<<grid, block>>>(
        w_quant_ptr, w_packed_ptr,
        static_cast<int>(N),
        static_cast<int>(K)
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "prepare_weights_cuda_kernel launch failed: ",
                cudaGetErrorString(err));

    // Return weight scale as same dtype/device as weight
    return std::make_tuple(w_scale.to(weight.dtype()), w_packed);
}