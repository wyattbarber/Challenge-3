#include <pybind11/pybind11.h>
#include "neuralnet.hpp"
namespace py = pybind11;
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>


PYBIND11_MODULE(neuralnet, m){
    m.doc() ="CIS 678 Challenge #3 C++ backend";

    py::enum_<ActivationFunc>(m, "ActivationFunctions")
        .value("ReLU", ActivationFunc::ReLU)
        .value("Sigmoid", ActivationFunc::Sigmoid)
        .export_values();

    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<std::vector<size_t>, std::vector<ActivationFunc>>())
        .def("forwardPass", &NeuralNetwork::forwardPass, "Performs one forward pass through the model",
            py::call_guard<py::gil_scoped_release, py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("train", &NeuralNetwork::train, "Performs backpropagation on a set of training data",
            py::call_guard<py::gil_scoped_release, py::scoped_ostream_redirect, py::scoped_estream_redirect>());
}
