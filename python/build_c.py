"""Build the qengine._C torch extension in place (python/qengine/_C*.so). Called by scripts/build.sh.
Only the BF16 decode kernels; csrc/gemv_w4* (qengine._w4) is built separately."""
import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).resolve().parent
CSRC = HERE.parent / "csrc"
arch = os.environ.get("ARCH", "sm_120")
SRC = ["bindings.cpp", "basic.cu", "gdn.cu", "attn.cu"]

os.chdir(HERE)
sys.argv = [sys.argv[0], "build_ext", "--inplace", "--build-temp", str(HERE.parent / "build" / "ext_c")]
setup(
    name="qengine_c",
    ext_modules=[CUDAExtension(
        "qengine._C", [str(CSRC / s) for s in SRC],
        extra_compile_args={"cxx": ["-O3", "-std=c++20"],
                            "nvcc": ["-O3", "-std=c++20", f"-arch={arch}", "-lineinfo",
                                     "--expt-relaxed-constexpr"]})],
    cmdclass={"build_ext": BuildExtension},
)
