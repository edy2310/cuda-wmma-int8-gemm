/*
    PyTorch CUDA implementation for the WMMA INT8 GEMM binding.
    Allocates the FP16 output tensor, resolves raw pointers, and launches the CUDA kernel on
    PyTorch's current CUDA stream.
*/

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "gemm_kernel.cuh"

torch::Tensor wmma_int8_gemm_cuda(torch::Tensor A,
                                  torch::Tensor B_colmajor,
                                  double scaleA,
                                  torch::Tensor scaleB) {
    // Resolve matrix sizes from input tensors.
    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B_colmajor.size(0);

    // Allocate FP16 output on the same device as A.
    auto C = torch::zeros({M, N}, A.options().dtype(torch::kFloat16));

    // Extract raw device pointers for kernel launch.
    const int8_t* A_ptr = A.data_ptr<int8_t>();
    const int8_t* B_ptr = B_colmajor.data_ptr<int8_t>();
    const float* scaleB_ptr = scaleB.data_ptr<float>();
    __half* C_ptr = reinterpret_cast<__half*>(C.data_ptr<at::Half>());

    dim3 blockSize(128, 1, 1);
    dim3 gridSize((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    // Use PyTorch's current CUDA stream for proper synchronization.
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // Launch WMMA INT8 GEMM with per-channel dequantization into FP16.
    wmmaInt8GemmKernel<<<gridSize, blockSize, 0, stream>>>(
        A_ptr,
        B_ptr,
        C_ptr,
        static_cast<float>(scaleA),
        scaleB_ptr,
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        static_cast<int>(K),
        static_cast<int>(K),
        static_cast<int>(N));

    return C;
}
