from pathlib import Path
import platform

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

torch_lib_path = Path(torch.__file__).parent / "lib"
libomp_root = Path("/opt/homebrew/opt/libomp")
libomp_include = libomp_root / "include"
libomp_lib = libomp_root / "lib"

# Select architecture-specific ukernel source at build time.
arch = platform.machine().lower()
if arch in ("arm64", "aarch64"):
    ukernel_src = "kernel/arm/ukernel.cpp"
else:
    ukernel_src = "kernel/x86/ukernel.cpp"

setup(
    name="bitlinear",
    ext_modules=[
        CppExtension(
            name="bitlinear",
            sources=["kernel/bitlinear_cpu.cpp", ukernel_src],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-march=native",
                    "-Xpreprocessor",
                    "-fopenmp",
                    f"-I{libomp_include}",
                ],
            },
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