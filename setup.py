from pathlib import Path
import os
import platform
import sys

from setuptools import setup

# Try to import torch - if not available, we'll handle it gracefully
# This allows setup.py to be parsed even if torch isn't installed yet
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # We'll import these later when they're actually needed
    # For now, we'll define the extension structure
    BuildExtension = None
    CppExtension = None
    CUDAExtension = None

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8") if (this_directory / "README.md").exists() else ""

# Get torch lib path if available
if TORCH_AVAILABLE:
    torch_lib_path = Path(torch.__file__).parent / "lib"
    # Resolve to absolute path and ensure it exists
    if torch_lib_path.exists():
        torch_lib_path = torch_lib_path.resolve()
    else:
        torch_lib_path = None
else:
    torch_lib_path = None

# Allow forcing a CPU-only build regardless of CUDA availability so local
# rebuilds can skip NVCC when iterating on CPU kernels.
force_cpu_build = bool(int(os.environ.get("BITLINEAR_FORCE_CPU", "0")))

# Check if CUDA is available (and not explicitly disabled)
if TORCH_AVAILABLE:
    cuda_available = (not force_cpu_build) and torch.cuda.is_available()
else:
    # If torch isn't available yet, default to CPU-only build
    # The actual CUDA check will happen when setup runs after torch is installed
    cuda_available = False

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
extra_link_args = []
library_dirs = []
if torch_lib_path and torch_lib_path.exists():
    # Add library directory for linking
    library_dirs.append(str(torch_lib_path))
    # On macOS, rpath should work, but we also ensure library_dirs is set
    if platform.system() == "Darwin":
        # Use absolute path in rpath for macOS
        extra_link_args.append(f"-Wl,-rpath,{torch_lib_path}")
        # Also add @rpath fallback
        extra_link_args.append("-Wl,-rpath,@loader_path")
    else:
        # Linux
        extra_link_args.append(f"-Wl,-rpath,{torch_lib_path}")

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

# Always create the extension - it will be built during installation
# The extension is created here but built by BuildExtension during setup
# If torch isn't available yet, we need to import it now or defer extension creation
if not TORCH_AVAILABLE:
    # Try to import torch again - it might be available now via setup_requires
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
        TORCH_AVAILABLE = True
        if torch_lib_path is None:
            torch_lib_path = Path(torch.__file__).parent / "lib"
            if torch_lib_path.exists():
                torch_lib_path = torch_lib_path.resolve()
                if platform.system() == "Darwin":
                    extra_link_args.insert(0, f"-Wl,-rpath,{torch_lib_path}")
                    library_dirs.insert(0, str(torch_lib_path))
                else:
                    extra_link_args.insert(0, f"-Wl,-rpath,{torch_lib_path}")
                    library_dirs.insert(0, str(torch_lib_path))
            else:
                torch_lib_path = None
        # Re-check CUDA availability
        cuda_available = (not force_cpu_build) and torch.cuda.is_available()
    except ImportError:
        # If torch still isn't available, we can't proceed
        # This should not happen if build-system.requires in pyproject.toml is working
        # The C++ files include <torch/extension.h> so torch headers are required
        raise RuntimeError(
            "PyTorch is required to build this package because the C++ source files "
            "include <torch/extension.h>. PyTorch should be installed automatically "
            "via build-system.requires in pyproject.toml. If you see this error, "
            "please ensure you're using a PEP 517-compatible installer (pip >= 19.0) "
            "or install PyTorch manually: pip install torch>=1.13.0"
        )

# Now create the extension with torch available
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
        library_dirs=library_dirs,
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
        library_dirs=library_dirs,
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
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Oliver",
    author_email="oliver@example.com",
    url="https://github.com/oliver/bitlinear",
    license="MIT",
    py_modules=["bitlinear"],  # Include bitlinear.py as a module
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension} if BuildExtension else {},
    python_requires=">=3.7",
    # Note: install_requires is specified in pyproject.toml [project] dependencies
    # to avoid conflicts. Build dependencies are in [build-system] requires.
    # Torch is required there for compilation since C++ files include <torch/extension.h>
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,  # Important for extensions
    keywords="pytorch neural-networks quantization binary-networks cuda bitlinear",
    include_package_data=True,  # Include files specified in MANIFEST.in
)
