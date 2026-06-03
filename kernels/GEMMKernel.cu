/*
    INT8 WMMA GEMM using Tensor Cores, shared-memory tiling, and vectorized loads.
    Computes C = A * B with A in row-major and B in column-major.
    This requires a GPU with Tensor Core INT8 support.
*/

#include <cuda_runtime.h>
#include <mma.h>

#include "gemm_kernel.cuh"

// Shared-memory padding: keep stride a multiple of 16 bytes for WMMA alignment.
constexpr int WMMA_PAD = 16;
constexpr int WMMA_STRIDE = WMMA_K + WMMA_PAD;

// CUDA kernel: each block computes one 16x16 output tile using one warp's WMMA ops.
// The block uses all threads to cooperatively load shared tiles with vectorized loads.
__global__ void wmmaInt8GemmKernel(const int8_t* __restrict__ A,
                                  const int8_t* __restrict__ B,
                                  __half* __restrict__ C,
                                  float scaleA, const float* __restrict__ scaleB,
                                  int M, int N, int K,
                                  int lda, int ldb, int ldc) {
    // Shared tiles with padding to reduce bank conflicts while keeping alignment.
    // Stride is 32 bytes (16 + 16 padding), which satisfies WMMA alignment constraints.
    __align__(16) __shared__ int8_t As[WMMA_M * WMMA_STRIDE];
    __align__(16) __shared__ int8_t Bs[WMMA_N * WMMA_STRIDE];
    __align__(16) __shared__ int32_t Cs[WMMA_M * WMMA_N];

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;

    const int rowStart = blockRow * WMMA_M;
    const int colStart = blockCol * WMMA_N;

    // Each warp performs WMMA on its own 16x16 sub-tile within the block tile.
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t,
                           nvcuda::wmma::row_major>
        aFrag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t,
                           nvcuda::wmma::col_major>
        bFrag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int>
        cFrag;
    nvcuda::wmma::fill_fragment(cFrag, 0);

    // Loop over K tiles.
    for (int tileK = 0; tileK < (K + WMMA_K - 1) / WMMA_K; ++tileK) {
        const int kStart = tileK * WMMA_K;

        // Vectorized loads of A: each of the first 16 threads loads one full row (16 int8s).
        if (tid < WMMA_M) {
            const int row = rowStart + tid;
            const int8_t* aRow = A + row * lda + kStart;
            int8_t* aTileRow = As + tid * WMMA_STRIDE;

            if (row < M && (kStart + WMMA_K) <= K &&
                (reinterpret_cast<uintptr_t>(aRow) % 16) == 0 &&
                (reinterpret_cast<uintptr_t>(aTileRow) % 16) == 0) {
                // Aligned 16-byte load.
                const int4* src = reinterpret_cast<const int4*>(aRow);
                reinterpret_cast<int4*>(aTileRow)[0] = src[0];
            } else {
                // Scalar fallback for edges or misalignment.
                for (int i = 0; i < WMMA_K; ++i) {
                    const int col = kStart + i;
                    aTileRow[i] = (row < M && col < K) ? aRow[i] : 0;
                }
            }
        }

        // Vectorized loads of B (column-major): each of the first 16 threads loads one column.
        if (tid < WMMA_N) {
            const int col = colStart + tid;
            const int8_t* bCol = B + col * ldb + kStart;
            // Column-major tile in shared memory with padded stride.
            int8_t* bTileCol = Bs + tid * WMMA_STRIDE;

            if (col < N && (kStart + WMMA_K) <= K &&
                (reinterpret_cast<uintptr_t>(bCol) % 16) == 0 &&
                (reinterpret_cast<uintptr_t>(bTileCol) % 16) == 0) {
                const int4* src = reinterpret_cast<const int4*>(bCol);
                reinterpret_cast<int4*>(bTileCol)[0] = src[0];
            } else {
                for (int i = 0; i < WMMA_K; ++i) {
                    const int row = kStart + i;
                    bTileCol[i] = (col < N && row < K) ? bCol[i] : 0;
                }
            }
        }

        // Ensure the shared tiles are ready for the warp.
        __syncthreads();

        if (warpId == 0) {
            // Load shared tiles into WMMA fragments and perform Tensor Core MMA.
            nvcuda::wmma::load_matrix_sync(aFrag, As, WMMA_STRIDE);
            nvcuda::wmma::load_matrix_sync(bFrag, Bs, WMMA_STRIDE);
            nvcuda::wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }

        __syncthreads();
    }

    // Store the accumulator fragment to shared memory, then write to global with bounds checks.
    if (warpId == 0) {
        nvcuda::wmma::store_matrix_sync(Cs, cFrag, WMMA_N, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < WMMA_M * WMMA_N; idx += blockDim.x) {
        const int r = idx / WMMA_N;
        const int c = idx % WMMA_N;
        const int globalRow = rowStart + r;
        const int globalCol = colStart + c;

        if (globalRow < M && globalCol < N) {
            const int32_t acc = Cs[r * WMMA_N + c];
            const float dequantScale = scaleA * scaleB[globalCol];
            C[globalRow * ldc + globalCol] = __float2half(static_cast<float>(acc) * dequantScale);
        }
    }
}