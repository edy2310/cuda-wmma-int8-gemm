# WMMA INT8 GEMM
This project implements a WMMA-based INT8 GEMM kernel for CUDA and PyTorch. It is designed as a compact, production-style example of GPU kernel engineering, systems integration, correctness validation, and performance analysis.

## Demonstrated Capabilities

- **Kernel Design**: WMMA-based INT8 GEMM with 16x16x16 tiling, int32 accumulation, and FP16 output.
- **Systems Integration**: PyTorch C++/CUDA extension with a minimal Python API and strict runtime checks.
- **Performance Analysis**: throughput, latency, and roofline benchmarking against PyTorch's INT8 path.
- **Correctness Engineering**: reference validation, edge-case coverage, determinism checks, and invalid-input tests.
- **Production Deployment Readiness**: clean packaging, explicit tensor/layout contracts, and a drop-in API for larger inference stacks.

## Architecture & Design Decisions

### Execution Flow

```text
Python
  -> gemm_api.wmma_int8_gemm
  -> C++ binding
  -> CUDA launcher
  -> WMMA kernel
  -> FP16 output
```

The flow is intentionally small and explicit so each layer can be inspected independently during debugging, profiling, or integration work.

### Memory Layout

- **A**: `int8`, row-major, contiguous, shape `[M, K]`
- **B**: `int8`, column-major, contiguous transpose, shape `[N, K]`
- **scaleA**: scalar `float`
- **scaleB**: `float32` CUDA tensor, shape `[N]`
- **Output**: `FP16`, shape `[M, N]`

The column-major `B` layout matches WMMA access patterns and avoids hidden transposes inside the kernel.

### Engineering Trade-offs

- Shared-memory staging gives predictable reuse, even if it leaves some occupancy on the table.
- `int4` vector loads are used when alignment allows; scalar fallback keeps edge cases correct.
- Accumulation stays in `int32` until the final dequantization step to preserve precision.
- The current kernel is a baseline by design: conservative enough to validate, profile, and improve systematically.

## Performance Results & Interpretation

Benchmarks use fixed square matrices (`256, 512, 1024, 2048`) with 10 warmup iterations and 50 timed iterations. The current baseline is measured against PyTorch's INT8 matmul path.

### Throughput

| Matrix Size | Custom Kernel (TFLOPS) | cuBLAS INT8 (TFLOPS) | Speedup vs cuBLAS INT8 (x) |
|-----------:|------------------------:|---------------------:|---------------------------:|
| 256 | 1.69 | 1.28 | 1.32x |
| 512 | 2.23 | 6.21 | 0.36x |
| 1024 | 2.69 | 14.04 | 0.19x |
| 2048 | 3.42 | 20.17 | 0.17x |

#### Plot

![Throughput](benchmarks/results/throughput.png)

#### Interpretation

The kernel is competitive at 256 and then falls behind as the matrices grow. That is a useful interview signal: the implementation works, the Tensor Core path is real, and the scaling gap points directly to missing production-grade optimizations such as deeper tiling, pipelining, and better warp scheduling.

### Roofline

#### Plot

![Roofline](benchmarks/results/roofline.png)

#### Interpretation

The custom kernel sits below the compute roof, which means there is still headroom in memory movement and execution efficiency. In practice, this says the current design is valid but not yet fully optimized for large-shape inference workloads.

### Latency

| Matrix Size | Custom Kernel (ms) | cuBLAS INT8 (ms) | Latency Ratio vs cuBLAS INT8 (x) |
|-----------:|-------------------:|----------------:|---------------------------------:|
| 256 | 0.024 | 0.033 | 0.74x |
| 512 | 0.150 | 0.055 | 2.73x |
| 1024 | 1.011 | 0.194 | 5.21x |
| 2048 | 4.812 | 0.837 | 5.75x |

#### Plot

![Latency](benchmarks/results/latency.png)

#### Interpretation

Latency tells the same story as throughput: the kernel can win on small inputs, but steady-state efficiency becomes the bottleneck as size increases. That is exactly the kind of result I want in a portfolio piece, because it shows I can measure a limitation, explain it, and turn it into an optimization plan.

## Correctness & Validation

The test suite compares the kernel against a PyTorch INT8 reference with identical dequantization. It covers normal shapes, tile-boundary sizes, random non-square stress, scale extremes, determinism, and invalid-input checks.

Single-command test run:

```bash
python tests/test_gemm_kernel.py
```

## Quick Start

### Requirements

- NVIDIA GPU with Tensor Core INT8 support (SM75+ recommended)
- CUDA-enabled PyTorch install
- Compatible CUDA toolkit and driver

### Build

```bash
python setup.py build_ext --inplace
```

### Example Usage

```python
import torch
import gemm_api

A = torch.randint(-128, 127, (64, 64), device="cuda", dtype=torch.int8)
B = torch.randint(-128, 127, (64, 64), device="cuda", dtype=torch.int8).t().contiguous()
scaleB = torch.full((64,), 0.2, device="cuda", dtype=torch.float32)

C = gemm_api.wmma_int8_gemm(A, B, 0.1, scaleB)
print(C.shape, C.dtype)
```

## Optimization Roadmap

The current kernel is intentionally a strong baseline, not the final form.

- Add multi-warp blocks and warp specialization to raise occupancy.
- Use `cp.async` double buffering to overlap global memory movement with Tensor Core compute.
- Explore split-K and larger tile shapes to improve arithmetic intensity on large matrices.
- Fuse epilogue work such as scaling, bias, or activation to reduce write bandwidth.
- Autotune tile size, stage count, and launch configuration per GPU generation.
- Extend the implementation into a serving backend once the kernel profile is stable.
