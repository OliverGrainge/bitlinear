from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

torch_lib_path = Path(torch.__file__).parent / "lib"
libomp_root = Path("/opt/homebrew/opt/libomp")
libomp_include = libomp_root / "include"
libomp_lib = libomp_root / "lib"

setup(
    name='bitlinear',
    ext_modules=[
        CppExtension(
            name='bitlinear',
            sources=['kernel/bitlinear_cache_dot.cpp'],
            extra_compile_args={
                'cxx': [
                    '-O3',
                    '-march=native',
                    '-Xpreprocessor',
                    '-fopenmp',
                    f"-I{libomp_include}",
                ],
            },
            extra_link_args=[
                f"-Wl,-rpath,{torch_lib_path}",
                f"-Wl,-rpath,{libomp_lib}",
                f"-L{libomp_lib}",
                '-lomp',
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)