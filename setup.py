from pathlib import Path
import os
import platform
import subprocess

from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension as TorchBuildExtension,
    CppExtension,
    CUDAExtension,
)
import torch


ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""


def get_torch_lib_path() -> Path | None:
    lib_path = Path(torch.__file__).parent / "lib"
    return lib_path.resolve() if lib_path.exists() else None


def get_openmp_paths():
    system = platform.system()
    if system == "Darwin":
        root = Path("/opt/homebrew/opt/libomp")
        return root / "include", root / "lib"
    else:
        inc = Path("/opt/homebrew/opt/libomp/include")
        lib = Path("/opt/homebrew/opt/libomp/lib")
        return (inc if inc.exists() else None), (lib if lib.exists() else None)


def has_cuda() -> bool:
    force_cpu = bool(int(os.environ.get("BITLINEAR_FORCE_CPU", "0")))
    if force_cpu:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def make_extra_compile_args(libomp_include):
    cxx_flags = ["-O3", "-march=native"]
    system = platform.system()

    if system == "Darwin":
        cxx_flags += ["-Xpreprocessor", "-fopenmp"]
        if libomp_include:
            cxx_flags.append(f"-I{libomp_include}")
    else:
        cxx_flags.append("-fopenmp")
        if libomp_include:
            cxx_flags.append(f"-I{libomp_include}")

    return {"cxx": cxx_flags}


def make_extra_link_args(torch_lib_path, libomp_lib):
    args = []
    system = platform.system()

    # Torch rpath
    if torch_lib_path:
        torch_lib_path = str(torch_lib_path)
        args.append(f"-Wl,-rpath,{torch_lib_path}")
        if system == "Darwin":
            args.append("-Wl,-rpath,@loader_path")

    # OpenMP
    if libomp_lib:
        libomp_lib = str(libomp_lib)
        args += [f"-Wl,-rpath,{libomp_lib}", f"-L{libomp_lib}", "-lomp"]

    return args


def make_cuda_flags():
    # Keep your more elaborate sm list/arch detection here if you want.
    supported_arches = ["70", "75", "80", "86", "89", "90"]
    fallback_ptx_arch = "90"

    env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env_arch_list:
        arch_list = [a.strip() for a in env_arch_list.split(";") if a.strip()]
    else:
        arch_list = supported_arches.copy()
        try:
            major, minor = torch.cuda.get_device_capability()
            detected = f"{major}{minor}"
            if detected not in arch_list:
                arch_list.append(detected)
        except Exception:
            pass

    nvcc_flags = ["-O3", "--use_fast_math"]
    for arch in arch_list:
        if arch.startswith("compute_"):
            sm = arch.split("_", 1)[1]
            nvcc_flags.append(f"-gencode=arch=compute_{sm},code=compute_{sm}")
        else:
            nvcc_flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")

    # Always include PTX for highest supported arch
    if not any(f"code=compute_{fallback_ptx_arch}" in f for f in nvcc_flags):
        nvcc_flags.append(
            f"-gencode=arch=compute_{fallback_ptx_arch},code=compute_{fallback_ptx_arch}"
        )

    return nvcc_flags


class BuildExtension(TorchBuildExtension):
    """Single place to fix macOS rpaths (if needed)."""

    def build_extension(self, ext):
        super().build_extension(ext)

        system = platform.system()
        if system != "Darwin":
            return

        torch_lib_path = get_torch_lib_path()
        if not torch_lib_path:
            return

        ext_path = self.get_ext_fullpath(ext.name)
        if not os.path.exists(ext_path):
            return

        try:
            subprocess.run(
                ["install_name_tool", "-add_rpath", str(torch_lib_path), ext_path],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"✓ Fixed rpath for {ext_path}")
        except subprocess.CalledProcessError as e:
            # Best-effort; if it already has the rpath, ignore.
            otool_result = subprocess.run(
                ["otool", "-l", ext_path],
                capture_output=True,
                text=True,
            )
            if str(torch_lib_path) in otool_result.stdout:
                print(f"✓ rpath already set for {ext_path}")
            else:
                print(f"⚠ Could not add rpath to {ext_path}: {e.stderr}")
        except FileNotFoundError:
            print("⚠ install_name_tool not found; rpath may be incorrect")


def get_extensions():
    torch_lib_path = get_torch_lib_path()
    libomp_include, libomp_lib = get_openmp_paths()

    extra_compile_args = make_extra_compile_args(libomp_include)
    extra_link_args = make_extra_link_args(torch_lib_path, libomp_lib)

    cpu_sources = [
        "csrc/bitlinear.cpp",
        "csrc/cpu/bitlinear_cpu.cpp",
    ]

    define_macros = []
    if has_cuda():
        define_macros.append(("WITH_CUDA", None))
        sources = cpu_sources + ["csrc/cuda/bitlinear_cuda.cu"]
        extra_compile_args["nvcc"] = make_cuda_flags()
        ext_cls = CUDAExtension
        print("=== Building BitLinear with CUDA support ===")
    else:
        sources = cpu_sources
        ext_cls = CppExtension
        print("=== Building BitLinear (CPU only) ===")

    ext = ext_cls(
        name="_bitlinear",
        sources=sources,
        include_dirs=["csrc", "csrc/cpu", "csrc/cuda"],
        library_dirs=[str(torch_lib_path)] if torch_lib_path else [],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    )
    return [ext]


setup(
    name="bitlinear",
    version="0.1.0",
    description="Efficient BitLinear implementation with CPU and CUDA support",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Oliver",
    author_email="oliver@example.com",
    url="https://github.com/oliver/bitlinear",
    license="MIT",
    py_modules=["bitlinear"],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
    include_package_data=True,
)