from pathlib import Path
import os
import platform

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

torch_lib_path = Path(torch.__file__).parent / "lib"

# Allow forcing a CPU-only build regardless of CUDA availability so local
# rebuilds can skip NVCC when iterating on CPU kernels.
force_cpu_build = bool(int(os.environ.get("BITLINEAR_FORCE_CPU", "0")))

# Check if CUDA is available (and not explicitly disabled)
cuda_available = (not force_cpu_build) and torch.cuda.is_available()

# OpenMP paths - macOS uses homebrew, Linux typically uses system libraries
if platform.system() == "Darwin":
    libomp_root = Path("/opt/homebrew/opt/libomp")
    libomp_include = libomp_root / "include"
    libomp_lib = libomp_root / "lib"
else:
    # Linux - OpenMP is typically in system paths, but check common locations
    libomp_include = (
        Path("/opt/homebrew/opt/libomp/include")
        if Path("/opt/homebrew/opt/libomp/include").exists()
        else None
    )
    libomp_lib = (
        Path("/opt/homebrew/opt/libomp/lib")
        if Path("/opt/homebrew/opt/libomp/lib").exists()
        else None
    )

# Source files
cpu_sources = [
    "csrc/bitlinear.cpp",  # Dispatcher
    "csrc/cpu/bitlinear_cpu.cpp",  # CPU implementation
]

cuda_sources = [
    "csrc/cuda/bitlinear_cuda.cu",
]

# Compiler flags for C++
extra_cxx_flags = [
    "-O3",
    "-march=native",
]

if platform.system() == "Darwin":
    extra_cxx_flags.extend(
        [
            "-Xpreprocessor",
            "-fopenmp",
            f"-I{libomp_include}",
        ]
    )
else:
    # Linux - use standard -fopenmp flag
    extra_cxx_flags.append("-fopenmp")
    if libomp_include and libomp_include.exists():
        extra_cxx_flags.append(f"-I{libomp_include}")

# Link arguments
extra_link_args = [f"-Wl,-rpath,{torch_lib_path}"]

if platform.system() == "Darwin":
    extra_link_args.extend(
        [
            f"-Wl,-rpath,{libomp_lib}",
            f"-L{libomp_lib}",
            "-lomp",
        ]
    )
else:
    if libomp_lib and libomp_lib.exists():
        extra_link_args.extend([f"-Wl,-rpath,{libomp_lib}", f"-L{libomp_lib}", "-lomp"])

# Define macros
define_macros = []

# Create extension based on CUDA availability
if cuda_available:
    # CUDA is available - build with CUDA support
    define_macros.append(("WITH_CUDA", None))

    # CUDA architectures we can safely emit SASS for with this toolkit
    supported_arches = ["70", "75", "80", "86", "89", "90"]
    fallback_ptx_arch = "90"
    detected_arch = None

    try:
        major, minor = torch.cuda.get_device_capability()
        detected_arch = f"{major}{minor}"
    except Exception:
        detected_arch = None

    # Warn users when running on GPUs newer than the toolkit supports.
    if detected_arch and detected_arch not in supported_arches:
        print(
            f"WARNING: Detected GPU capability sm_{detected_arch} but CUDA "
            "12.0 can only emit binaries up to sm_90. "
            f"Falling back to PTX for compute_{fallback_ptx_arch}."
        )

    # Allow overriding via TORCH_CUDA_ARCH_LIST for reproducibility
    env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env_arch_list:
        arch_list = [a.strip() for a in env_arch_list.split(";") if a.strip()]
    else:
        arch_list = supported_arches.copy()
        if (
            detected_arch
            and detected_arch in supported_arches
            and detected_arch not in arch_list
        ):
            arch_list.append(detected_arch)

    extra_nvcc_flags = ["-O3", "--use_fast_math"]
    for arch in arch_list:
        if arch.startswith("compute_"):
            # Allow users to request PTX-only targets via env var
            sm = arch.split("_", 1)[1]
            extra_nvcc_flags.append(f"-gencode=arch=compute_{sm},code=compute_{sm}")
        else:
            extra_nvcc_flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")

    # Always include PTX for the highest supported arch so newer GPUs can JIT.
    if not any(
        flag.endswith(f"code=compute_{fallback_ptx_arch}") for flag in extra_nvcc_flags
    ):
        extra_nvcc_flags.append(
            f"-gencode=arch=compute_{fallback_ptx_arch},code=compute_{fallback_ptx_arch}"
        )

    extension = CUDAExtension(
        name="_bitlinear",
        sources=cpu_sources + cuda_sources,
        include_dirs=["csrc", "csrc/cpu", "csrc/cuda"],
        extra_compile_args={
            "cxx": extra_cxx_flags,
            "nvcc": extra_nvcc_flags,
        },
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    )

    print("=" * 60)
    print("Building BitLinear with CUDA support")
    print("=" * 60)
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Sources: {len(cpu_sources)} CPU + {len(cuda_sources)} CUDA files")
    print("=" * 60)
else:
    # CUDA not available - build CPU-only version
    extension = CppExtension(
        name="_bitlinear",
        sources=cpu_sources,
        include_dirs=["csrc", "csrc/cpu"],
        extra_compile_args={"cxx": extra_cxx_flags},
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    )

    print("=" * 60)
    print("Building BitLinear with CPU-only support")
    print("=" * 60)
    print("CUDA not available - skipping GPU kernels")
    print(f"Sources: {len(cpu_sources)} CPU files")
    print("=" * 60)

setup(
    name="bitlinear",
    version="0.1.0",
    description="Efficient BitLinear implementation with CPU and CUDA support",
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.13.0",
    ],
)
