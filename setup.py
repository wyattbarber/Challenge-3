from glob import glob
from setuptools import setup
import distutils.cmd
import subprocess
from pybind11.setup_helpers import Pybind11Extension, build_ext

pybindext = Pybind11Extension(
        "neuralnet",
        sorted(glob("src/*.cpp")),
        cxx_std=17,
        extra_compile_args=['/openmp', '-IC:\\msys64\\mingw64\\include\\eigen3'],
        extra_link_args=['-lopenmp']
)

ext_modules = [
    pybindext,
]

setup(
    cmdclass={
        "build_ext": build_ext
        }, 
    ext_modules=ext_modules
)