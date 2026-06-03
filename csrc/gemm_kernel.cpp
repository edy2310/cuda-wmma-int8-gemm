/*
    PyTorch C++ binding entrypoint.
    Validates inputs and exposes the WMMA INT8 GEMM API to Python via pybind11.
*/

#include <torch/extension.h>

torch::Tensor wmma_int8_gemm_cuda(torch::Tensor A,
                                  torch::Tensor B_colmajor,
                                  double scaleA,
                                  torch::Tensor scaleB);

torch::Tensor wmma_int8_gemm(torch::Tensor A,
                             torch::Tensor B_colmajor,
                             double scaleA,
                             torch::Tensor scaleB) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor.");
    TORCH_CHECK(B_colmajor.is_cuda(), "B must be a CUDA tensor.");
    TORCH_CHECK(scaleB.is_cuda(), "scaleB must be a CUDA tensor.");
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8.");
    TORCH_CHECK(B_colmajor.dtype() == torch::kInt8, "B must be int8.");
    TORCH_CHECK(scaleB.dtype() == torch::kFloat32, "scaleB must be float32.");
    TORCH_CHECK(A.dim() == 2, "A must be 2D (M x K).");
    TORCH_CHECK(B_colmajor.dim() == 2, "B must be 2D (N x K) in column-major layout.");
    TORCH_CHECK(scaleB.dim() == 1, "scaleB must be 1D with length N.");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous.");
    TORCH_CHECK(B_colmajor.is_contiguous(), "B must be contiguous and stored as [N, K].");
    TORCH_CHECK(scaleB.is_contiguous(), "scaleB must be contiguous.");
    TORCH_CHECK(A.get_device() == B_colmajor.get_device(), "A and B must be on the same CUDA device.");
    TORCH_CHECK(A.get_device() == scaleB.get_device(), "scaleB must be on the same CUDA device as A.");
    TORCH_CHECK(A.size(1) == B_colmajor.size(1), "K dimension mismatch between A and B.");
    TORCH_CHECK(scaleB.size(0) == B_colmajor.size(0), "scaleB length must equal N (B.size(0)).");

    return wmma_int8_gemm_cuda(A, B_colmajor, scaleA, scaleB);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wmma_int8_gemm", &wmma_int8_gemm, "WMMA INT8 GEMM with per-channel dequant (FP16 output)");
}
