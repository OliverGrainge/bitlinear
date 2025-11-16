from pathlib import Path
import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

torch_lib_path = Path(torch.__file__).parent / "lib"
# OpenMP paths - macOS uses homebrew, Linux typically uses system libraries
import platform
if platform.system() == "Darwin":
    libomp_root = Path("/opt/homebrew/opt/libomp")
    libomp_include = libomp_root / "include"
    libomp_lib = libomp_root / "lib"
else:
    # Linux - OpenMP is typically in system paths, but check common locations
    libomp_include = Path("/opt/homebrew/opt/libomp/include") if Path("/opt/homebrew/opt/libomp/include").exists() else None
    libomp_lib = Path("/opt/homebrew/opt/libomp/lib") if Path("/opt/homebrew/opt/libomp/lib").exists() else None

# Optional build-time profiling for the CPU kernel. If the environment
# variable BITLINEAR_PROFILE is set to a non-empty value, we define the
# BITLINEAR_PROFILE macro in the C++ build. This will make the kernel print
# simple timing information for key sections (quantization, loading tiles,
# inner matmul, writeback).
enable_profile = bool(os.environ.get("BITLINEAR_PROFILE"))

extra_cxx_flags = [
    "-O3",
    "-march=native",
]
if platform.system() == "Darwin":
    extra_cxx_flags.extend([
        "-Xpreprocessor",
        "-fopenmp",
        f"-I{libomp_include}",
    ])
else:
    # Linux - use standard -fopenmp flag
    extra_cxx_flags.append("-fopenmp")
    if libomp_include and libomp_include.exists():
        extra_cxx_flags.append(f"-I{libomp_include}")

if enable_profile:
    extra_cxx_flags.append("-DBITLINEAR_PROFILE=1")

setup(
    name="bitlinear",
    ext_modules=[
        CppExtension(
            name="bitlinear",
            # Include both the main CPU kernel and the ukernel implementations
            sources=["kernel/bitlinear_cpu.cpp", "kernel/ukernel.cpp"],
            include_dirs=["kernel"],
            extra_compile_args={"cxx": extra_cxx_flags},
            extra_link_args=[
                f"-Wl,-rpath,{torch_lib_path}",
            ] + ([
                f"-Wl,-rpath,{libomp_lib}",
                f"-L{libomp_lib}",
                "-lomp",
            ] if platform.system() == "Darwin" else (
                [f"-Wl,-rpath,{libomp_lib}", f"-L{libomp_lib}", "-lomp"] 
                if (libomp_lib and libomp_lib.exists()) 
                else []
            )),
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)