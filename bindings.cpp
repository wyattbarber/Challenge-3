#include "include/basic/Layer.hpp"
#include "include/basic/PySequence.hpp"
#include "include/basic/Compound.hpp"
#include "include/autoencoder/AutoEncoder.hpp"
#include "include/autoencoder/DeepAutoEncoder.hpp"
#include "include/training/Trainer.hpp"
#include "include/optimizers/Optimizer.hpp"
#include "include/convolutional/Conv2D.hpp"
#include <tuple>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>

namespace py = pybind11;
using namespace neuralnet;
using namespace optimization;

template<class T, typename... Ts>
void make_model(py::module m, const char* name)
{
    py::class_<T, DynamicModel<double>, std::shared_ptr<T>>(m, name)
    .def(py::init<Ts...>())
    .def("forward", static_cast<Eigen::VectorXd (T::*)(Eigen::VectorXd&)>(&T::forward), "Performs a forward pass through the model.");
}

PYBIND11_MODULE(neuralnet, m)
{
    m.doc() = "Various neural network implementations";

    py::class_<DynamicModel<double>, std::shared_ptr<DynamicModel<double>>>(m, "Model");

    make_model<DynamicBinder<double, Layer<double, ActivationFunc::Linear, OptimizerClass::Adam>>, size_t, size_t, double, double>(m, "Linear");
    make_model<DynamicBinder<double, Layer<double, ActivationFunc::ReLU, OptimizerClass::Adam>>, size_t, size_t, double, double>(m, "ReLU");
    make_model<DynamicBinder<double, Layer<double, ActivationFunc::Sigmoid, OptimizerClass::Adam>>, size_t, size_t, double, double>(m, "Sigmoid");
    make_model<DynamicBinder<double, Layer<double, ActivationFunc::TanH, OptimizerClass::Adam>>, size_t, size_t, double, double>(m, "TanH");
    make_model<DynamicBinder<double, Layer<double, ActivationFunc::SoftMax, OptimizerClass::Adam>>, size_t, size_t, double, double>(m, "SoftMax");

    make_model<DynamicBinder<double, AutoEncoder<double, ActivationFunc::ReLU, OptimizerClass::None>>, size_t, size_t, double, double>(m, "AutoEncoder");
    make_model<DynamicBinder<double, DeepAutoEncoder<double, ActivationFunc::TanH, ActivationFunc::ReLU, ActivationFunc::Sigmoid, OptimizerClass::Adam>>,
        std::vector<size_t>, double, double>(m, "DeepAutoEncoder");

    make_model<DynamicBinder<double, PySequence<double>>, std::vector<std::shared_ptr<DynamicModel<double>>>>(m, "Sequence");

    py::class_<training::Trainer<DynamicBinder<double, PySequence<double>>>>(m, "Trainer")
        .def(py::init<
            DynamicBinder<double, PySequence<double>>&,
            std::vector<DynamicModel<double>::InputType>&,
            std::vector<DynamicModel<double>::OutputType>&
        >())
        .def("train", &training::Trainer<DynamicBinder<double, PySequence<double>>>::train, "Trains a model", py::return_value_policy::automatic);

    m.def("convolve", [](Eigen::Tensor<double,3> in, size_t in_channels, size_t out_channels){
        return Convolution2D<double, 5, OptimizerClass::None>(in_channels, out_channels).forward(in);
    });
}
