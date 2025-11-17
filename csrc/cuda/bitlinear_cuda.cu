// csrc/cuda/bitlinear_cuda.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ----------------- matmul kernel -----------------

// Tunable tile sizes; TILE_K must be divisible by 4
constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 64;  // 64 elements along K

__global__ void bitlinear_cuda_kernel(
    const float* __restrict__ x_scale_ptr,   // [M]
    const int8_t* __restrict__ x_quant_ptr,  // [M, K]
    const float* __restrict__ w_scale_ptr,   // scalar
    const uint8_t* __restrict__ w_packed_ptr,// [N, K/4], row-major over N
    const float* __restrict__ bias_ptr,      // [N]
    float* __restrict__ y_ptr,               // [M, N]
    int M, int K, int N) {

    // Block tile origin in output space
    const int block_row = blockIdx.x * TILE_M;
    const int block_col = blockIdx.y * TILE_N;

    const int local_row = threadIdx.x; // 0..TILE_M-1
    const int local_col = threadIdx.y; // 0..TILE_N-1

    const int row = block_row + local_row;
    const int col = block_col + local_col;

    if (row >= M || col >= N) {
        return;
    }

    // Shared memory tiles:
    //  - sX: [TILE_M, TILE_K] int8 activations
    //  - sW: [TILE_N, TILE_K] int8 weights (decoded to {-1,0,1})
    __shared__ int8_t sX[TILE_M][TILE_K];
    __shared__ int8_t sW[TILE_N][TILE_K];

    int32_t acc_i = 0;

    // Loop over K dimension in chunks of TILE_K
    for (int kk = 0; kk < K; kk += TILE_K) {
        // 1) Load activations tile into shared memory
        // Each thread strides over K inside the tile to fill sX[local_row][*]
        for (int k_inner = local_col; k_inner < TILE_K; k_inner += blockDim.y) {
            const int gk = kk + k_inner;
            int8_t val = 0;
            if (row < M && gk < K) {
                val = x_quant_ptr[row * K + gk];
            }
            sX[local_row][k_inner] = val;
        }

        // 2) Load weights tile into shared memory, decoding from packed 2-bit
        // Each thread strides over K inside the tile to fill sW[local_col][*]
        for (int k_inner = local_row; k_inner < TILE_K; k_inner += blockDim.x) {
            const int gk = kk + k_inner;
            int8_t w_val = 0;

            if (col < N && gk < K) {
                const int pack_idx    = gk / 4;          // which byte along K/4
                const int idx_in_pack = gk & 0x3;        // 0..3
                const uint8_t w_byte =
                    w_packed_ptr[col * (K / 4) + pack_idx];

                const int shift = 6 - 2 * idx_in_pack;
                w_val = static_cast<int8_t>((w_byte >> shift) & 0x03) - 1;
            }

            sW[local_col][k_inner] = w_val;
        }

        __syncthreads();

        // 3) Compute on this K tile
        // All entries are valid or zero-padded, so we can safely do TILE_K steps
        #pragma unroll
        for (int k_inner = 0; k_inner < TILE_K; ++k_inner) {
            const int8_t xv = sX[local_row][k_inner];
            const int8_t wv = sW[local_col][k_inner];
            acc_i += static_cast<int32_t>(xv) * static_cast<int32_t>(wv);
        }

        __syncthreads();
    }

    const float x_scale = x_scale_ptr[row]; // per-row
    const float w_scale = w_scale_ptr[0];   // global
    const float bias    = bias_ptr[col];

    const float acc_f = static_cast<float>(acc_i);
    y_ptr[row * N + col] = acc_f * x_scale * w_scale + bias;
}


// ----------------- C++ wrapper -----------------

torch::Tensor bitlinear_cuda(torch::Tensor x,
                             torch::Tensor w_scale,
                             torch::Tensor w_packed,
                             c10::optional<torch::Tensor> bias_opt,
                             float eps) {
    TORCH_CHECK(x.is_cuda(),        "x must be a CUDA tensor");
    TORCH_CHECK(w_scale.is_cuda(),  "w_scale must be a CUDA tensor");
    TORCH_CHECK(w_packed.is_cuda(), "w_packed must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2,       "x must be 2D [M, K]");
    TORCH_CHECK(w_packed.dim() == 2,"w_packed must be 2D [N, K/4]");

    TORCH_CHECK(x.dtype() == torch::kFloat32 ||
                x.dtype() == torch::kFloat16 ||
                x.dtype() == torch::kBFloat16,
                "x must be floating (f32/f16/bf16)");

    TORCH_CHECK(w_scale.numel() == 1,
                "w_scale must be a scalar tensor (numel==1)");

    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));
    const int N = static_cast<int>(w_packed.size(0));

    TORCH_CHECK(K % 4 == 0, "K must be divisible by 4");
    TORCH_CHECK(static_cast<int64_t>(w_packed.size(1)) == K / 4,
                "w_packed has inconsistent second dimension (expected K/4)");

    // ---- Activation quantization ----
    // x_scale: [M, 1] then squeezed to [M]
    auto x_fp32  = x.to(torch::kFloat32);
    // per-row max magnitude, avoid division by zero
    auto x_scale = std::get<0>(x_fp32.abs().max(-1, true)).clamp_min(eps) / 127.0f;
    auto x_quant = (x_fp32 / x_scale)
                       .round()
                       .clamp(-128, 127)
                       .to(torch::kInt8)
                       .contiguous();

    x_scale = x_scale.squeeze(-1).contiguous();  // [M]

    // ---- Data pointers ----
    const int8_t*  x_quant_ptr  = x_quant.data_ptr<int8_t>();
    const float*   x_scale_ptr  = x_scale.data_ptr<float>();

    auto w_scale_f32 = w_scale.to(torch::kFloat32).contiguous();
    const float* w_scale_ptr = w_scale_f32.data_ptr<float>();

    TORCH_CHECK(w_packed.dtype() == torch::kUInt8,
                "w_packed must be uint8");
    const uint8_t* w_packed_ptr = w_packed.contiguous().data_ptr<uint8_t>();

    auto opts_f32 = x.options().dtype(torch::kFloat32).device(x.device());
    auto y = torch::empty({M, N}, opts_f32).contiguous();
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
    // 16x16 keeps occupancy reasonable without oversubscribing registers
    dim3 block(TILE_M, TILE_N); // (16, 16)
    dim3 grid(
        (M + TILE_M - 1) / TILE_M,
        (N + TILE_N - 1) / TILE_N
    );

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

__global__ void prepare_weights_cuda_kernel(
    const int8_t* __restrict__ w_quant_ptr, // [N, K] in {-1,0,1}+1 => {0,1,2}
    uint8_t* __restrict__ w_packed_ptr,     // [N, K/4]
    int N,
    int K) {

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) {
        return;
    }

    const int row_offset_in  = row * K;
    const int row_offset_out = row * (K / 4);

    // K is divisible by 4
    #pragma unroll 4
    for (int k = 0; k < K; k += 4) {
        const uint8_t w0 = static_cast<uint8_t>(w_quant_ptr[row_offset_in + k + 0]);
        const uint8_t w1 = static_cast<uint8_t>(w_quant_ptr[row_offset_in + k + 1]);
        const uint8_t w2 = static_cast<uint8_t>(w_quant_ptr[row_offset_in + k + 2]);
        const uint8_t w3 = static_cast<uint8_t>(w_quant_ptr[row_offset_in + k + 3]);

        w_packed_ptr[row_offset_out + (k / 4)] =
            static_cast<uint8_t>((w0 << 6) | (w1 << 4) | (w2 << 2) | w3);
    }
}


std::tuple<torch::Tensor, torch::Tensor> prepare_weights_cuda(
    torch::Tensor weight,  // [N, K]
    float eps,
    std::string /*quant_type*/) {

    TORCH_CHECK(weight.is_cuda(),       "weight must be a CUDA tensor");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.dim() == 2,      "weight must be 2D [N, K]");

    const int64_t N64 = weight.size(0);
    const int64_t K64 = weight.size(1);

    TORCH_CHECK(K64 % 4 == 0,
                "K must be divisible by 4 for this packing (got K=", K64, ")");

    const int N = static_cast<int>(N64);
    const int K = static_cast<int>(K64);

    auto opts_f32 = weight.options().dtype(torch::kFloat32);
    auto opts_u8  = weight.options().dtype(torch::kUInt8);
    auto opts_i8  = weight.options().dtype(torch::kInt8);

    auto w_fp32 = weight.to(opts_f32);  // [N, K] float32

    // Single global scale for ternary weights; you might want something more stable,
    // but keep semantics identical to original: mean(|w|)
    auto w_scale = w_fp32.abs().mean().clamp_min(eps);

    // Quantize to {-1,0,1}, then shift to {0,1,2} for packing
    auto w_quant = ((w_fp32 / w_scale)
                        .round()
                        .clamp(-1, 1) + 1)
                       .to(opts_i8)
                       .contiguous();

    auto w_packed = torch::empty({N64, K64 / 4}, opts_u8).contiguous();

    const int8_t* w_quant_ptr  = w_quant.data_ptr<int8_t>();
    uint8_t*      w_packed_ptr = w_packed.data_ptr<uint8_t>();

    // 256 threads per block is a reasonable default; N is along x
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    prepare_weights_cuda_kernel<<<grid, block>>>(
        w_quant_ptr,
        w_packed_ptr,
        N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "prepare_weights_cuda_kernel launch failed: ",
                cudaGetErrorString(err));

    // Return weight scale as same dtype/device as weight
    return std::make_tuple(w_scale.to(weight.dtype()), w_packed);
}