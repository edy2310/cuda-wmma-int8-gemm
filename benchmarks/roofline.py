from __future__ import annotations

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import torch

# Add the repo root and python/ to sys.path so the local extension module is importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

import gemm_api

# Fixed benchmark configuration for square matrices.
DEFAULT_SIZES = [256, 512, 1024, 2048]
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 50
PEAK_FP16_TFLOPS = 65.0
PEAK_BW_GBPS = 320.0


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


def tflops(m: int, n: int, k: int, ms: float) -> float:
    # GEMM flop count: 2 * M * N * K. Convert ms to seconds.
    return (2.0 * m * n * k) / (ms * 1e-3) / 1e12


def arithmetic_intensity(m: int, n: int, k: int, bytes_a: int, bytes_b: int, bytes_c: int) -> float:
    # FLOPs divided by bytes moved (A + B + C write).
    flops = 2.0 * m * n * k
    bytes_moved = m * k * bytes_a + n * k * bytes_b + m * n * bytes_c
    return flops / bytes_moved


def int8_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Use PyTorch's int8 GEMM path (cuBLASLt) when available.
    if hasattr(torch, "_int_mm"):
        return torch._int_mm(A, B)
    if hasattr(torch.ops.aten, "_int_mm"):
        return torch.ops.aten._int_mm(A, B)
    raise RuntimeError("INT8 GEMM is not available in this PyTorch build.")


def run_roofline_points(sizes: list[tuple[int, int, int]], warmup: int, iters: int) -> list[dict]:
    # Measure attainable TFLOPS and compute arithmetic intensity for each implementation.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    points = []
    torch.manual_seed(0)

    for (M, N, K) in sizes:
        A_int8 = torch.randint(-128, 127, (M, K), device="cuda", dtype=torch.int8)
        B_colmajor = torch.randint(-128, 127, (N, K), device="cuda", dtype=torch.int8)

        scaleA = 0.1
        scaleB = torch.full((N,), 0.2, device="cuda", dtype=torch.float32)

        B_int8 = B_colmajor.t().contiguous()

        kernel_ms = benchmark_op(
            lambda: gemm_api.wmma_int8_gemm(A_int8, B_colmajor, scaleA, scaleB),
            warmup,
            iters,
        )
        cublas_int8_ms = benchmark_op(lambda: int8_matmul(A_int8, B_int8), warmup, iters)

        points.append(
            {
                "size": M,
                "kernel_tflops": tflops(M, N, K, kernel_ms),
                "kernel_ai": arithmetic_intensity(M, N, K, 1, 1, 2),
                "cublas_int8_tflops": tflops(M, N, K, cublas_int8_ms),
                "cublas_int8_ai": arithmetic_intensity(M, N, K, 1, 1, 4),
            }
        )

    return points


def plot_roofline(points: list[dict], output_path: Path) -> None:
    # Plot arithmetic intensity vs attainable TFLOPS for each implementation.
    kernel_ai = [p["kernel_ai"] for p in points]
    kernel_tf = [p["kernel_tflops"] for p in points]
    int8_ai = [p["cublas_int8_ai"] for p in points]
    int8_tf = [p["cublas_int8_tflops"] for p in points]

    plt.figure(figsize=(7, 4))
    max_ai = min(1000.0, max(kernel_ai + int8_ai) * 1.2)
    ai_line = [1e-3, max_ai]
    bw_tflops = [(PEAK_BW_GBPS * 1e9 * ai) / 1e12 for ai in ai_line]

    # Roofline ceilings: bandwidth and compute limits.
    plt.plot(ai_line, bw_tflops, linestyle="--", color="gray", label="Peak Bandwidth")
    plt.hlines(PEAK_FP16_TFLOPS, xmin=ai_line[0], xmax=ai_line[1], linestyle="--", color="orange", label="Peak FP16")
    plt.plot(kernel_ai, kernel_tf, marker="o", label="Custom Kernel")
    plt.plot(int8_ai, int8_tf, marker="o", label="cuBLAS INT8")

    plt.xlim(0.0, 1000.0)
    plt.xlabel("Arithmetic Intensity (FLOPs / byte)")
    plt.ylabel("Attainable Throughput (TFLOPS)")
    plt.title("Roofline Model (TFLOPS vs Arithmetic Intensity)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    sizes = sizes_as_triples(DEFAULT_SIZES)
    points = run_roofline_points(sizes, DEFAULT_WARMUP, DEFAULT_ITERS)
    plot_roofline(points, ROOT / "benchmarks" / "roofline.png")


if __name__ == "__main__":
    main()
