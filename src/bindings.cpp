#include <pybind11/pybind11.h>
#include "layer.hpp"
#include "validation.hpp"
namespace py = pybind11;
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>


PYBIND11_MODULE(neuralnet, m){
    m.doc() ="CIS 678 Challenge #3 C++ backend";

    py::enum_<activation::ActivationFunc>(m, "ActivationFunctions")
        .value("ReLU", activation::ActivationFunc::ReLU)
        .value("Sigmoid", activation::ActivationFunc::Sigmoid)
        .value("SoftMax", activation::ActivationFunc::SoftMax)
        .export_values();

    py::class_<Network>(m, "Network")
        .def(py::init<std::vector<int>, std::vector<activation::ActivationFunc>>())
        .def("train", &Network::train, "Performs backpropagation on a set of training data",
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("forward", &Network::forward, "Performs a single forward pass through the model",
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    m.def("test", &test, "Performs cross validation on one model and reports its percent error");
}
