#include <pybind11/pybind11.h>
#include "neuralnet.hpp"
namespace py = pybind11;
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>


PYBIND11_MODULE(neuralnet, m){
    m.doc() ="CIS 678 Challenge #3 C++ backend";

    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<std::vector<size_t>>())
        .def("forwardPass", &NeuralNetwork::forwardPass, "Performs one forward pass through the model")
        .def("backprop", &NeuralNetwork::backprop, "Performs backpropagation on a set of training data");
}