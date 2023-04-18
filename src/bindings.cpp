#include "layer.hpp"
#include "validation.hpp"
#include "autoencoder.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
namespace py = pybind11;


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

    py::class_<Adam, Network>(m, "AdamNetwork")
        .def(py::init<std::vector<int>, std::vector<activation::ActivationFunc>>())
        .def("train", &Adam::train, "Performs backpropagation on a set of training data",
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("forward", &Adam::forward, "Performs a single forward pass through the model",
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    py::class_<Autoencoder>(m, "Autoencoder")
        .def(py::init<size_t, size_t>())
        .def("train", &Autoencoder::train, "Performs backpropagation on a set of training data",
           py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("encode", &Autoencoder::encode, "Transforms a datapoint to latent space",
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("decode", &Autoencoder::decode, "Generates an approximation from a latent representation",
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    
    py::class_<DeepAutoencoder>(m, "DeepAutoencoder")
        .def(py::init<std::vector<size_t>>())
        .def("train", &DeepAutoencoder::train, "Performs backpropagation on a set of training data",
           py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());


    m.def("test", &test, "Performs cross validation on one model and reports its percent error");
    m.def("test_layers", &test_dimensions, "Performs cross validation over models with different layer count and sizes",
        py::call_guard<py::gil_scoped_release>());
}
