from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent

"""
Builds the PyTorch CUDA extension that exposes the WMMA INT8 GEMM kernel to Python.
This uses torch.utils.cpp_extension so the build inherits PyTorch's include paths,
CUDA toolchain configuration, and ABI settings.
Both the C++ and NVCC compilation paths use -O3 to enable aggressive optimization
for host code and device kernels.
"""
setup(
    name="gemm_kernel",
    py_modules=["gemm_api"],
    ext_modules=[
        CUDAExtension(
            name="gemm_kernel_ext",
            sources=[
                "csrc/gemm_kernel.cpp",
                "csrc/gemm_kernel_cuda.cu",
                "kernels/GEMMKernel.cu",
            ],
            include_dirs=[str(ROOT / "kernels")],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
