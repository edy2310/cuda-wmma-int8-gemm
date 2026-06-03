from __future__ import annotations
import torch
import gemm_kernel_ext


def wmma_int8_gemm(
    A: torch.Tensor,
    B_colmajor: torch.Tensor,
    scaleA: float,
    scaleB: torch.Tensor,
) -> torch.Tensor:
    """
    Run the WMMA INT8 GEMM kernel with per-channel dequantization.

    Args:
        A: int8 CUDA tensor shaped [M, K], row-major.
        B_colmajor: int8 CUDA tensor shaped [N, K], column-major layout.
        scaleA: per-tensor scale for A.
        scaleB: float32 CUDA tensor shaped [N], per-column scale for B.

    Returns:
        FP16 CUDA tensor shaped [M, N].
    """
    return gemm_kernel_ext.wmma_int8_gemm(A, B_colmajor, scaleA, scaleB)
