#include "neuralnet.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
namespace py = pybind11;


PYBIND11_MODULE(neuralnet, m){
    m.doc() ="Various neural network implementations";

    py::enum_<activation::ActivationFunc>(m, "ActivationFunctions")
        .value("ReLU", activation::ActivationFunc::ReLU)
        .value("Sigmoid", activation::ActivationFunc::Sigmoid)
        .value("SoftMax", activation::ActivationFunc::SoftMax)
        .export_values();

    py::class_<Network>(m, "Network")
        .def(py::init<std::vector<int>, std::vector<activation::ActivationFunc>>())
        .def("train",  py::overload_cast<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>, double, int>(&Network::train), "Performs backpropagation on a set of training data",
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("train",  py::overload_cast<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>, double, int, double, double>(&Network::train), "Performs backpropagation on a set of training data",
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("forward", &Network::forward, "Performs a single forward pass through the model",
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());


    py::class_<Autoencoder>(m, "Autoencoder")
        .def(py::init<size_t, size_t>())
        .def("train", py::overload_cast<Eigen::MatrixXd, double, int>(&Autoencoder::train), "Performs backpropagation on a set of training data",
           py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("train", py::overload_cast<Eigen::MatrixXd, double, int, double, double>(&Autoencoder::train), "Performs backpropagation on a set of training data using the Adam algorithm",
           py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("encode", &Autoencoder::encode, "Transforms a datapoint to latent space",
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("decode", &Autoencoder::decode, "Generates an approximation from a latent representation",
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    
    py::class_<DeepAutoencoder>(m, "DeepAutoencoder")
        .def(py::init<std::vector<size_t>>())
        .def("train", py::overload_cast<Eigen::MatrixXd, double, int>(&DeepAutoencoder::train), "Performs backpropagation on a set of training data",
           py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("train", py::overload_cast<Eigen::MatrixXd, double, int, double, double>(&DeepAutoencoder::train), "Performs backpropagation on a set of training data using the Adam algorithm",
           py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("encode", &DeepAutoencoder::encode, "Transforms a datapoint to latent space",
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("decode", &DeepAutoencoder::decode, "Generates an approximation from a latent representation",
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    py::class_<CoupledAutoencoder>(m, "CoupledAutoencoder")
        .def(py::init<std::vector<size_t>, std::vector<size_t>, size_t>())
        .def("train", py::overload_cast<Eigen::MatrixXd, Eigen::MatrixXd, double, int, double>(&CoupledAutoencoder::train), "Performs backpropagation on a set of training data",
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("train", py::overload_cast<Eigen::MatrixXd, Eigen::MatrixXd, double, int, double, double, double>(&CoupledAutoencoder::train), "Performs backpropagation on a set of training data using the Adam algorithm",
           py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("encode", &CoupledAutoencoder::encode, "Transforms a datapoint to latent space",
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("decode", &CoupledAutoencoder::decode, "Generates an approximation from a latent representation",
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    py::class_<VariationalAutoencoder>(m, "VariationalAutoencoder")
        .def(py::init<std::vector<size_t>>())
        .def("train", py::overload_cast<Eigen::MatrixXd, double, int, int>(&VariationalAutoencoder::train), "Performs backpropagation on a set of training data",
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("train", py::overload_cast<Eigen::MatrixXd, double, int, int, double, double>(&VariationalAutoencoder::train), "Performs backpropagation on a set of training data using the Adam algorithm",
           py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("encode", &VariationalAutoencoder::encode, "Transforms a datapoint to latent space",
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("decode", &VariationalAutoencoder::decode, "Generates an approximation from a latent representation",
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    
}
