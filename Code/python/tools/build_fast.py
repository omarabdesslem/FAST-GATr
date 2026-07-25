"""
Standalone fast-build script. Bypasses setup.py.
Run from Code/python/ directory:
    python tools/build_fast.py build_ext --inplace --force
"""
import sys
from pathlib import Path

import pybind11
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

PYTHON_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PYTHON_DIR.parent
CPP = CODE_DIR / "cpp" / "optimized"

FLAGS = [
    "/std:c++17",
    "/O2",
    "/Ob3",
    "/fp:fast",
    "/DNDEBUG",
    "/arch:AVX2",
    "/GL",
]
LINK = ["/LTCG"]

def ext(name, sources):
    return CppExtension(
        name,
        sources=[str(CPP / s) for s in sources],
        include_dirs=[pybind11.get_include(), str(CPP)],
        extra_compile_args=list(FLAGS),
        extra_link_args=list(LINK),
    )

setup(
    name="ops",
    ext_modules=[
        ext("ops.linear",    ["linear.cpp"]),
        ext("ops.dual",      ["dual.cpp", "linear.cpp"]),
        ext("ops.attention", ["attention.cpp", "linear.cpp"]),
        ext("ops.norm",      ["norm.cpp"]),
    ],
    cmdclass={"build_ext": BuildExtension},
)
