import os
import sys

import torch


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "python"))

from gemm_api import wmma_int8_gemm  # noqa: E402


def make_inputs(M: int, K: int, N: int, device: torch.device):
    """Create int8 inputs plus dequant scales."""
    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
    B_rowmajor = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
    # Kernel expects B in column-major layout as a contiguous [N, K] tensor.
    B_colmajor = B_rowmajor.t().contiguous()
    scaleA = float(torch.rand(1, device=device).item()) * 0.25 + 0.01
    scaleB = torch.rand(N, device=device, dtype=torch.float32) * 0.25 + 0.01
    return A, B_rowmajor, B_colmajor, scaleA, scaleB


def reference_gemm(A: torch.Tensor,
                   B_rowmajor: torch.Tensor,
                   scaleA: float,
                   scaleB: torch.Tensor) -> torch.Tensor:
    """Reference GEMM using PyTorch (cuBLAS for int8) then dequantize."""
    acc = torch.matmul(A.int(), B_rowmajor.int())
    dequant = acc.float() * (scaleA * scaleB).view(1, -1)
    return dequant.half()


def assert_close(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    if not torch.allclose(actual, expected, rtol=1e-2, atol=1e-2):
        max_diff = (actual - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch (max diff {max_diff})")


def assert_raises(fn, contains: str) -> None:
    try:
        fn()
    except RuntimeError as exc:
        if contains and contains not in str(exc):
            raise AssertionError(f"Expected error containing '{contains}', got: {exc}") from exc
        return
    raise AssertionError("Expected RuntimeError was not raised")


def test_happy_paths(device: torch.device) -> None:
    # Typical sizes including non-multiples of 16.
    cases = [
        (16, 16, 16),
        (32, 48, 64),
        (17, 19, 23),
        (31, 64, 15),
    ]
    for M, K, N in cases:
        A, B_rowmajor, B_colmajor, scaleA, scaleB = make_inputs(M, K, N, device)
        C_kernel = wmma_int8_gemm(A, B_colmajor, scaleA, scaleB)
        C_ref = reference_gemm(A, B_rowmajor, scaleA, scaleB)
        assert_close(C_kernel, C_ref, f"happy_path_{M}x{K}x{N}")


def test_bad_paths(device: torch.device) -> None:
    # Validate input checks and error messages.
    A, B_rowmajor, B_colmajor, scaleA, scaleB = make_inputs(16, 16, 16, device)

    assert_raises(
        lambda: wmma_int8_gemm(A.float(), B_colmajor, scaleA, scaleB),
        "A must be int8",
    )

    assert_raises(
        lambda: wmma_int8_gemm(A, B_colmajor.float(), scaleA, scaleB),
        "B must be int8",
    )

    assert_raises(
        lambda: wmma_int8_gemm(A, B_colmajor, scaleA, scaleB.half()),
        "scaleB must be float32",
    )

    assert_raises(
        lambda: wmma_int8_gemm(A.cpu(), B_colmajor, scaleA, scaleB),
        "A must be a CUDA tensor",
    )

    A_nc = A.t()
    B_for_nc = torch.randint(-128, 127, (A_nc.size(1), 16), dtype=torch.int8, device=device)
    B_for_nc = B_for_nc.t().contiguous()
    scaleB_nc = torch.rand(B_for_nc.size(0), device=device, dtype=torch.float32)
    assert_raises(
        lambda: wmma_int8_gemm(A_nc, B_for_nc, scaleA, scaleB_nc),
        "A must be contiguous",
    )

    B_nc = B_rowmajor.t()
    assert_raises(
        lambda: wmma_int8_gemm(A, B_nc, scaleA, scaleB),
        "B must be contiguous",
    )

    scaleB_short = torch.rand(scaleB.size(0) - 1, device=device, dtype=torch.float32)
    assert_raises(
        lambda: wmma_int8_gemm(A, B_colmajor, scaleA, scaleB_short),
        "scaleB length must equal N",
    )


def test_edge_sizes(device: torch.device) -> None:
    # Cover tile boundary edges around 16x16x16.
    cases = [
        (1, 1, 1),
        (15, 15, 15),
        (16, 16, 16),
        (17, 17, 17),
        (31, 31, 31),
        (32, 32, 32),
    ]
    for M, K, N in cases:
        A, B_rowmajor, B_colmajor, scaleA, scaleB = make_inputs(M, K, N, device)
        C_kernel = wmma_int8_gemm(A, B_colmajor, scaleA, scaleB)
        C_ref = reference_gemm(A, B_rowmajor, scaleA, scaleB)
        assert_close(C_kernel, C_ref, f"edge_size_{M}x{K}x{N}")


def test_random_stress(device: torch.device) -> None:
    # Randomly sample non-square sizes to stress general shapes.
    torch.manual_seed(123)
    for _ in range(5):
        M = int(torch.randint(1, 65, (1,), device="cpu").item())
        K = int(torch.randint(1, 65, (1,), device="cpu").item())
        N = int(torch.randint(1, 65, (1,), device="cpu").item())
        A, B_rowmajor, B_colmajor, scaleA, scaleB = make_inputs(M, K, N, device)
        C_kernel = wmma_int8_gemm(A, B_colmajor, scaleA, scaleB)
        C_ref = reference_gemm(A, B_rowmajor, scaleA, scaleB)
        assert_close(C_kernel, C_ref, f"random_{M}x{K}x{N}")


def test_scale_extremes(device: torch.device) -> None:
    # Check very small and larger scales for numerical stability.
    A, B_rowmajor, B_colmajor, _, _ = make_inputs(32, 32, 32, device)
    for scaleA, scaleB_val in [(1e-3, 1e-3), (0.5, 0.5)]:
        scaleB = torch.full((B_colmajor.size(0),), scaleB_val, device=device, dtype=torch.float32)
        C_kernel = wmma_int8_gemm(A, B_colmajor, scaleA, scaleB)
        C_ref = reference_gemm(A, B_rowmajor, scaleA, scaleB)
        assert_close(C_kernel, C_ref, f"scale_extreme_{scaleA}_{scaleB_val}")


def test_determinism(device: torch.device) -> None:
    # Same input should produce identical output across repeated runs.
    A, _, B_colmajor, scaleA, scaleB = make_inputs(32, 32, 32, device)
    C_first = wmma_int8_gemm(A, B_colmajor, scaleA, scaleB)
    C_second = wmma_int8_gemm(A, B_colmajor, scaleA, scaleB)
    assert_close(C_first, C_second, "determinism")


def test_constant_inputs(device: torch.device) -> None:
    # Use predictable inputs to sanity-check outputs.
    M, K, N = 16, 16, 16
    A = torch.ones((M, K), dtype=torch.int8, device=device)
    B_rowmajor = torch.ones((K, N), dtype=torch.int8, device=device)
    B_colmajor = B_rowmajor.t().contiguous()
    scaleA = 0.1
    scaleB = torch.full((N,), 0.2, device=device, dtype=torch.float32)
    C_kernel = wmma_int8_gemm(A, B_colmajor, scaleA, scaleB)
    C_ref = reference_gemm(A, B_rowmajor, scaleA, scaleB)
    assert_close(C_kernel, C_ref, "constant_inputs")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run these tests.")

    device = torch.device("cuda")

    results = []
    failures = 0

    for name, fn in [
        ("happy_paths", lambda: test_happy_paths(device)),
        ("bad_paths", lambda: test_bad_paths(device)),
        ("edge_sizes", lambda: test_edge_sizes(device)),
        ("random_stress", lambda: test_random_stress(device)),
        ("scale_extremes", lambda: test_scale_extremes(device)),
        ("determinism", lambda: test_determinism(device)),
        ("constant_inputs", lambda: test_constant_inputs(device)),
    ]:
        try:
            fn()
            results.append((name, "PASS", ""))
        except Exception as exc:
            failures += 1
            results.append((name, "FAIL", str(exc)))

    print("TEST RESULTS")
    for name, status, message in results:
        line = f"- {name}: {status}"
        if message:
            line += f" ({message})"
        print(line)

    if failures:
        raise RuntimeError(f"{failures} test group(s) failed")


if __name__ == "__main__":
    main()
