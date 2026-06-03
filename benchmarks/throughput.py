from __future__ import annotations
from pathlib import Path
import sys
import time

import torch
import matplotlib.pyplot as plt

# Add the repo root and python/ to sys.path so the local extension module is importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

import gemm_api

# Fixed benchmark configuration for square matrices.
DEFAULT_SIZES = [256, 512, 1024, 2048]
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 50


def sizes_as_triples(sizes: list[int]) -> list[tuple[int, int, int]]:
    # Convert size list into (M, N, K) triples for square GEMMs.
    return [(size, size, size) for size in sizes]


def benchmark_op(fn, warmup: int, iters: int) -> float:
    # Warm up to avoid measuring one-time CUDA setup costs.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # Time the steady-state execution and return average seconds.
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def tflops(m: int, n: int, k: int, seconds: float) -> float:
    # GEMM flop count: 2 * M * N * K.
    return (2.0 * m * n * k) / seconds / 1e12


def int8_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Use PyTorch's int8 GEMM path (cuBLASLt) when available.
    if hasattr(torch, "_int_mm"):
        return torch._int_mm(A, B)
    if hasattr(torch.ops.aten, "_int_mm"):
        return torch.ops.aten._int_mm(A, B)
    raise RuntimeError("INT8 GEMM is not available in this PyTorch build.")


def run_benchmarks(sizes: list[tuple[int, int, int]], warmup: int, iters: int) -> list[dict]:
    # Prepare inputs and compare custom kernel vs cuBLAS INT8 for each size.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    results = []
    torch.manual_seed(0)

    for (M, N, K) in sizes:
        # Create synthetic int8 inputs and per-channel scales on the GPU.
        A_int8 = torch.randint(-128, 127, (M, K), device="cuda", dtype=torch.int8)
        B_colmajor = torch.randint(-128, 127, (N, K), device="cuda", dtype=torch.int8)

        scaleA = 0.1
        scaleB = torch.full((N,), 0.2, device="cuda", dtype=torch.float32)

        # Build int8 baseline in row-major for cuBLAS matmul.
        B_int8 = B_colmajor.t().contiguous()

        # Time the custom kernel and cuBLAS baselines.
        kernel_s = benchmark_op(
            lambda: gemm_api.wmma_int8_gemm(A_int8, B_colmajor, scaleA, scaleB),
            warmup,
            iters,
        )

        cublas_int8_s = benchmark_op(lambda: int8_matmul(A_int8, B_int8), warmup, iters)

        # Compute how close the kernel is to cuBLAS (ratio of TFLOPS).
        results.append(
            {
                "M": M,
                "N": N,
                "K": K,
                "kernel_tflops": tflops(M, N, K, kernel_s),
                "cublas_int8_tflops": tflops(M, N, K, cublas_int8_s),
                "speedup_vs_int8": tflops(M, N, K, kernel_s) / tflops(M, N, K, cublas_int8_s),
            }
        )

    return results


def format_markdown_table(results: list[dict]) -> str:
    # Emit a Markdown table ready to paste into Readme.md.
    lines = [
        "| Matrix Size | Custom Kernel (TFLOPS) | cuBLAS INT8 (TFLOPS) | Speedup vs cuBLAS INT8 (x) |",
        "|-----------:|--------------------:|--------------------:|---------------------------:|",
    ]
    for row in results:
        lines.append(
            f"| {row['M']} | {row['kernel_tflops']:.2f} | {row['cublas_int8_tflops']:.2f} | "
            f"{row['speedup_vs_int8']:.2f}x |"
        )
    return "\n".join(lines)


def plot_tflops(results: list[dict], output_path: Path) -> None:
    # Plot TFLOPS vs matrix size for each implementation.
    sizes = [row["M"] for row in results]
    kernel_tflops = [row["kernel_tflops"] for row in results]
    cublas_int8_tflops = [row["cublas_int8_tflops"] for row in results]

    plt.figure(figsize=(7, 4))
    plt.plot(sizes, kernel_tflops, marker="o", label="Custom Kernel")
    plt.plot(sizes, cublas_int8_tflops, marker="o", label="cuBLAS INT8")
    plt.xlabel("Matrix Size (square M=N=K)")
    plt.ylabel("TFLOPS")
    plt.title("GEMM TFLOPS vs Matrix Size")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    # Run the fixed-size benchmark sweep and print the Markdown table.
    sizes = sizes_as_triples(DEFAULT_SIZES)
    results = run_benchmarks(sizes, DEFAULT_WARMUP, DEFAULT_ITERS)
    print(format_markdown_table(results))
    plot_tflops(results, ROOT / "benchmarks" / "throughput.png")


if __name__ == "__main__":
    main()
