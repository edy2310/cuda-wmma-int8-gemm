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
    # Time the steady-state execution and return average milliseconds.
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / iters


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

        kernel_ms = benchmark_op(
            lambda: gemm_api.wmma_int8_gemm(A_int8, B_colmajor, scaleA, scaleB),
            warmup,
            iters,
        )
        cublas_int8_ms = benchmark_op(lambda: int8_matmul(A_int8, B_int8), warmup, iters)

        results.append(
            {
                "size": M,
                "kernel_ms": kernel_ms,
                "cublas_int8_ms": cublas_int8_ms,
            }
        )

    return results


def plot_latency(results: list[dict], output_path: Path) -> None:
    # Plot latency (ms) vs matrix size for each implementation.
    sizes = [row["size"] for row in results]
    kernel_ms = [row["kernel_ms"] for row in results]
    cublas_int8_ms = [row["cublas_int8_ms"] for row in results]

    plt.figure(figsize=(7, 4))
    plt.plot(sizes, kernel_ms, marker="o", label="Custom Kernel")
    plt.plot(sizes, cublas_int8_ms, marker="o", label="cuBLAS INT8")
    plt.xlabel("Matrix Size (square M=N=K)")
    plt.ylabel("Latency (ms)")
    plt.title("GEMM Latency vs Matrix Size")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def format_markdown_table(results: list[dict]) -> str:
    lines = [
        "| Matrix Size | Custom Kernel (ms) | cuBLAS INT8 (ms) | Latency Ratio vs cuBLAS INT8 (x) |",
        "|-----------:|-------------------:|----------------:|---------------------------------:|",
    ]
    for row in results:
        lines.append(
            f"| {row['size']} | {row['kernel_ms']:.3f} | {row['cublas_int8_ms']:.3f} | "
            f"{(row['kernel_ms'] / row['cublas_int8_ms']):.2f}x |"
        )
    return "\n".join(lines)


def main() -> None:
    sizes = sizes_as_triples(DEFAULT_SIZES)
    results = run_benchmarks(sizes, DEFAULT_WARMUP, DEFAULT_ITERS)
    print(format_markdown_table(results))
    plot_latency(results, ROOT / "benchmarks" / "latency.png")


if __name__ == "__main__":
    main()
