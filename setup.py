from pathlib import Path
import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

torch_lib_path = Path(torch.__file__).parent / "lib"
libomp_root = Path("/opt/homebrew/opt/libomp")
libomp_include = libomp_root / "include"
libomp_lib = libomp_root / "lib"

# Optional build-time profiling for the CPU kernel. If the environment
# variable BITLINEAR_PROFILE is set to a non-empty value, we define the
# BITLINEAR_PROFILE macro in the C++ build. This will make the kernel print
# simple timing information for key sections (quantization, loading tiles,
# inner matmul, writeback).
enable_profile = bool(os.environ.get("BITLINEAR_PROFILE"))

extra_cxx_flags = [
    "-O3",
    "-march=native",
    "-Xpreprocessor",
    "-fopenmp",
    f"-I{libomp_include}",
]

if enable_profile:
    extra_cxx_flags.append("-DBITLINEAR_PROFILE=1")

setup(
    name="bitlinear",
    ext_modules=[
        CppExtension(
            name="bitlinear",
            # ukernel implementations are now header-only (see kernel/ukernel.h),
            # so we only need the main CPU kernel translation unit here.
            sources=["kernel/bitlinear_cpu.cpp"],
            include_dirs=["kernel"],
            extra_compile_args={"cxx": extra_cxx_flags},
            extra_link_args=[
                f"-Wl,-rpath,{torch_lib_path}",
                f"-Wl,-rpath,{libomp_lib}",
                f"-L{libomp_lib}",
                "-lomp",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)