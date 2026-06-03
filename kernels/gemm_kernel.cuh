#pragma once

#include <cuda_fp16.h>
#include <cstdint>

// WMMA tile sizes for INT8 Tensor Cores.
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 1;

__global__ void wmmaInt8GemmKernel(const int8_t* __restrict__ A,
                                  const int8_t* __restrict__ B,
                                  __half* __restrict__ C,
                                  float scaleA, const float* __restrict__ scaleB,
                                  int M, int N, int K,
                                  int lda, int ldb, int ldc);
