from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "neuralnet",
        sorted(glob("src/*.cpp")),  # Sort source files for reproducibility
        cxx_std=17,
        extra_compile_args=['/openmp', '-IC:\\msys64\\mingw64\\include\\eigen3'],
        extra_link_args=['-lopenmp']
    ),
]

setup(
    name = "neuralnet",
    version="1.1.0",
    cmdclass={
        "build_ext": build_ext
        }, 
    ext_modules=ext_modules
)