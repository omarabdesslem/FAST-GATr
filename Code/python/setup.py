import os
import platform
import shlex
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import pybind11


PYTHON_DIR = Path(__file__).parent.resolve()
CODE_DIR = PYTHON_DIR.parent
CPP_IMPL = os.environ.get("CPP_IMPL", "optimized").strip().lower()
_VALID_CPP_IMPLS = {"baseline", "optimized", "optimized_for_cache", "simd"}
if CPP_IMPL not in _VALID_CPP_IMPLS:
    raise RuntimeError(f"CPP_IMPL must be one of {sorted(_VALID_CPP_IMPLS)}")

CPP_DIR = CODE_DIR / "cpp" / CPP_IMPL
OPS_DIR = PYTHON_DIR / "ops"
BUILD_TEMP_DIR = PYTHON_DIR / "build" / CPP_IMPL / "temp"

OPS_DIR.mkdir(exist_ok=True)
(OPS_DIR / "__init__.py").touch()
print(f"[setup.py] Building C++ implementation from {CPP_DIR}")

# to build with per-kernel __rdtsc cycle counters.
#   set CYCLE_COUNT=1 (Windows cmd) or export CYCLE_COUNT=1 (bash)
# before running this script. Default is off.
_CYCLE_COUNT_ENABLED = os.environ.get("CYCLE_COUNT", "") not in ("", "0")
_USE_CLANG = os.environ.get("USE_CLANG", "") not in ("", "0")
_IS_WIN = sys.platform == "win32"
_IS_MSVC = _IS_WIN and not _USE_CLANG
_IS_MAC = sys.platform == "darwin"
_IS_ARM = platform.machine().lower() in ("arm64", "aarch64")
if _IS_MAC and _IS_ARM:
    os.environ["ARCHFLAGS"] = "-arch arm64"
# Locate clang-cl from the VS 2022 BuildTools install.
_CLANG_CL = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\Llvm\x64\bin\clang-cl.exe"

if _IS_MSVC:
    _BASE_FLAGS = [
        "/std:c++17",
        "/O2",
        "/d2Qvec-",      # disable MSVC auto-vectorizer (no SSE/AVX gen from loops)
       # "/Ob3",         #  more aggressive inlining
       # "/fp:fast",     # fast-math
       # "/DNDEBUG",     # disable asserts and TORCH_CHECKs in hot paths
       # "/arch:AVX2",   # enable AVX2 and FMAs
       # "/GL",          # link-time optimizations
    ]
    # /LTCG is only useful when objects were compiled with /GL.
    _LINK_FLAGS = ["/LTCG"] if "/GL" in _BASE_FLAGS else []
    _CYCLE_FLAGS = ["/DCYCLE_COUNT"]
elif _IS_WIN and _USE_CLANG:
    # clang-cl with /clang: passthrough to give it the same flags as the Mac branch.
    # /O2 at the driver level prevents clang-cl from defaulting to /Od (no opt);
    # /clang:-O3 then bumps the underlying optimizer to -O3.
    _BASE_FLAGS = [
        "/O2",
        "/clang:-std=c++17",
        "/clang:-O3",
        "/clang:-ffast-math",
        "/clang:-DNDEBUG",
        "/clang:-funroll-loops",
        # No -march=native: keep ISA at the x86-64 baseline (SSE2 only). re-add later when optimizing
        "/clang:-fno-vectorize",       # disable loop vectorizer
        "/clang:-fno-slp-vectorize",   # disable SLP vectorizer
    ]
    _LINK_FLAGS = []
    _CYCLE_FLAGS = ["/DCYCLE_COUNT"]
else:
    _BASE_FLAGS = [
        "-std=c++17",
        "-O3",
        #"-g", when profiling so that profiling can symbolize the assembly.
        "-DNDEBUG",
        "-funroll-loops",
        #"-fno-omit-frame-pointer",  # ensure RDTSC cycles are accurate and not skewed by frame pointer omission enable when profiling and re-disable when not profiling
    ]
    if CPP_IMPL != "simd":
        _BASE_FLAGS += [
            "-fno-tree-vectorize",   # disable loop vectorizer
            "-fno-slp-vectorize",    # disable SLP vectorizer
        ]
    if _IS_MAC:
        # Xcode 26 libc++ marks some standard type traits as non-specializable.
        # torch 2.4.0 still specializes std::is_arithmetic in c10/util/strong_type.h.
        _BASE_FLAGS += [
            "-Wno-error=invalid-specialization",
            "-Wno-invalid-specialization",
        ]
    if _IS_ARM:
        if _IS_MAC:
            _BASE_FLAGS.append(f"-mcpu={os.environ.get('ASL_APPLE_MCPU', 'native')}")
        else:
            _BASE_FLAGS.append("-mcpu=native")
    else:
        _BASE_FLAGS.append("-march=native")
    _LINK_FLAGS = []
    _CYCLE_FLAGS = ["-DCYCLE_COUNT"]

_FLAG_OVERRIDE = os.environ.get("ASL_CXX_FLAGS", "").strip()
if _FLAG_OVERRIDE:
    _EXTRA_COMPILE_ARGS = shlex.split(_FLAG_OVERRIDE)
    print(f"[setup.py] ASL_CXX_FLAGS override: {_EXTRA_COMPILE_ARGS}")
else:
    _EXTRA_COMPILE_ARGS = list(_BASE_FLAGS)
if _CYCLE_COUNT_ENABLED:
    _EXTRA_COMPILE_ARGS += _CYCLE_FLAGS
    print("[setup.py] CYCLE_COUNT enabled - kernels will record __rdtsc cycles")


class BuildExt(BuildExtension):
    def initialize_options(self):
        super().initialize_options()
        self.build_lib = str(PYTHON_DIR)
        self.build_temp = str(BUILD_TEMP_DIR)

    def build_extensions(self):
        if _IS_WIN and _USE_CLANG:
            compiler = self.compiler
            if not compiler.initialized:
                compiler.initialize()
            if not os.path.isfile(_CLANG_CL):
                raise RuntimeError(f"clang-cl not found at {_CLANG_CL}")
            compiler.cc = _CLANG_CL
            print(f"[setup.py] USE_CLANG=1 - compiling with {_CLANG_CL}")
        super().build_extensions()


setup(
    name="ops",
    packages=["ops"],
    ext_modules=[
        CppExtension(
            "ops.linear",
            sources=[str(CPP_DIR / "linear.cpp")],
            include_dirs=[pybind11.get_include(), str(CPP_DIR)],
            extra_compile_args=list(_EXTRA_COMPILE_ARGS),
            extra_link_args=list(_LINK_FLAGS),
        ),
        CppExtension(
            "ops.dual",
            sources=[
                str(CPP_DIR / "dual.cpp"),
                str(CPP_DIR / "linear.cpp"),
            ],
            include_dirs=[pybind11.get_include(), str(CPP_DIR)],
            extra_compile_args=list(_EXTRA_COMPILE_ARGS),
            extra_link_args=list(_LINK_FLAGS),
        ),
        CppExtension(
            "ops.attention",
            sources=[
                str(CPP_DIR / "attention.cpp"),
                str(CPP_DIR / "linear.cpp"),
            ],
            include_dirs=[pybind11.get_include(), str(CPP_DIR)],
            extra_compile_args=list(_EXTRA_COMPILE_ARGS),
            extra_link_args=list(_LINK_FLAGS),
        ),
        CppExtension(
            "ops.norm",
            sources=[str(CPP_DIR / "norm.cpp")],
            include_dirs=[pybind11.get_include(), str(CPP_DIR)],
            extra_compile_args=list(_EXTRA_COMPILE_ARGS),
            extra_link_args=list(_LINK_FLAGS),
        ),
    ],
    cmdclass={"build_ext": BuildExt},
)
