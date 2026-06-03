/*
    Minimal C++ target for Nsight Compute profiling.
    Launches the WMMA INT8 GEMM kernel once with fixed sizes.
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <vector>

#include "../kernels/gemm_kernel.cuh"

int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    std::vector<int8_t> hA(M * K, 1);
    std::vector<int8_t> hB(N * K, 1);
    std::vector<float> hScaleB(N, 0.2f);
    const float scaleA = 0.1f;

    int8_t* dA = nullptr;
    int8_t* dB = nullptr;
    __half* dC = nullptr;
    float* dScaleB = nullptr;

    cudaMalloc(&dA, M * K * sizeof(int8_t));
    cudaMalloc(&dB, N * K * sizeof(int8_t));
    cudaMalloc(&dC, M * N * sizeof(__half));
    cudaMalloc(&dScaleB, N * sizeof(float));

    cudaMemcpy(dA, hA.data(), M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), N * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dScaleB, hScaleB.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(128, 1, 1);
    dim3 gridSize((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);

    wmmaInt8GemmKernel<<<gridSize, blockSize>>>(dA, dB, dC, scaleA, dScaleB, M, N, K, K, K, N);
    cudaDeviceSynchronize();

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dScaleB);

    return 0;
}
