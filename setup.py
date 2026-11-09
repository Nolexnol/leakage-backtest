"""Build script for the optional C++ extension (leakbt._fastbt).

`pip install -e .` compiles it; if no compiler is available the package still
works using the NumPy fallbacks in leakbt/portfolio.py.
"""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

ext_modules = [
    Pybind11Extension(
        "leakbt._fastbt",
        ["cpp/fastbt.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3"],
    )
]

setup(
    name="leakbt",
    version="0.1.0",
    description="Measuring how data leakage inflates backtested trading performance",
    packages=find_packages(exclude=("tests", "experiments")),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.10",
    install_requires=["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"],
    extras_require={"test": ["pytest"]},
)
