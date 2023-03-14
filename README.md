Challenge 3 Setup Instructions
==============================

Dependencies
------------

This project uses PyBind11 to compile python modules. It's C++ source is included as a submodule, run:
```
git submodule update --recursive
```
to make sure it is installed. To install PyBind11 python module, run:
```
pip install pybind11
```

Build
-----
To build the module, run:
```
pip install .
```
This will build and install the module using setup.py. If all works, you should be able to run:
```{py}
install neuralnet

model = neuralnet.NeuralNetwork()
```

in a notebook or other script.