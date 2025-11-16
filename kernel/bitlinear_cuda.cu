#include <cuda.h>
#include <cuda_runtime.h> 
#include <torch/extension.h> 


__global__ void bitlinear_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; 
    int col = blockIdx.y * blockDim.y + threadIdx.y; 

    if (row < M && col < N) {
        float sum = 0.0f; 
        for (int k = 0; k < K; k++) {
            sum += B[row * K + k] * B[col * K + k]; 
        }
        C[row * N + col] = sum; 
    }
}

torch::Tensor bitlinear_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0); 
    int N = B.size(0); 
    int K = A.size(1); 

    torch::Tensor C = torch::zeros({M, N}, torch::kFloat32); 

    dim3 block(32, 32); 
    dim3 grid((N + 31) / 32, (M + 31) / 32); 

    bitlinear_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K); 

    return C; 
}