#define PYBIND11_DETAILED_ERROR_MESSAGES

#include "basic/Layer.hpp"
#include "basic/PySequence.hpp"
// #include "basic/StaticPySequence.hpp"
#include "training/Trainer.hpp"
#include "optimizers/Optimizer.hpp"
#include <tuple>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>

namespace py = pybind11;
using namespace neuralnet;
using namespace optimization;

// auto model(bool optimized)
// {
//     auto s = StaticPySequence<784, 10, double>(Layer<784, 500, double, ActivationFunc::ReLU, OptimizerClass::Adam>(),
//                                              Layer<500, 300, double, ActivationFunc::ReLU, OptimizerClass::Adam>(),
//                                              Layer<300, 300, double, ActivationFunc::ReLU, OptimizerClass::Adam>(),
//                                              Layer<300, 100, double, ActivationFunc::ReLU, OptimizerClass::Adam>(),
//                                              Layer<100, 50, double, ActivationFunc::ReLU, OptimizerClass::Adam>(),
//                                              Layer<50, 10, double, ActivationFunc::SoftMax, OptimizerClass::Adam>());
//     return s;
// }

PYBIND11_MODULE(neuralnet, m)
{
    m.doc() = "Various neural network implementations";

    py::class_<ModelBase> basemodel(m, "_ModelBase");
    py::class_<Model<Eigen::Dynamic, Eigen::Dynamic, double>>(m, "Model", basemodel);

    py::class_<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::Linear, OptimizerClass::Adam>, Model<Eigen::Dynamic, Eigen::Dynamic, double>>(m, "Linear")
        .def("new", &makeModel<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::Linear, OptimizerClass::Adam>, size_t, size_t, double, double>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::Linear, OptimizerClass::Adam>::forward, "Performs a forward pass through the model.", py::return_value_policy::reference);
        
    py::class_<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::ReLU, OptimizerClass::Adam>, Model<Eigen::Dynamic, Eigen::Dynamic, double>>(m, "ReLU")
        .def("new", &makeModel<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::ReLU, OptimizerClass::Adam>, size_t, size_t, double, double>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::ReLU, OptimizerClass::Adam>::forward, "Performs a forward pass through the model.", py::return_value_policy::reference);
    py::class_<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::Sigmoid, OptimizerClass::Adam>, Model<Eigen::Dynamic, Eigen::Dynamic, double>>(m, "Sigmoid")
        .def("new", &makeModel<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::Sigmoid, OptimizerClass::Adam>, size_t, size_t, double, double>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::Sigmoid, OptimizerClass::Adam>::forward, "Performs a forward pass through the model.", py::return_value_policy::reference);
        
    py::class_<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::TanH, OptimizerClass::Adam>, Model<Eigen::Dynamic, Eigen::Dynamic, double>>(m, "TanH")
        .def("new", &makeModel<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::TanH, OptimizerClass::Adam>, size_t, size_t, double, double>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::TanH, OptimizerClass::Adam>::forward, "Performs a forward pass through the model.", py::return_value_policy::reference);
        
    py::class_<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::SoftMax, OptimizerClass::Adam>, Model<Eigen::Dynamic, Eigen::Dynamic, double>>(m, "SoftMax")
        .def("new", &makeModel<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::SoftMax, OptimizerClass::Adam>, size_t, size_t, double, double>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::SoftMax, OptimizerClass::Adam>::forward, "Performs a forward pass through the model.", py::return_value_policy::reference);
        
    py::class_<PySequence<Eigen::Dynamic, Eigen::Dynamic, double>, Model<Eigen::Dynamic, Eigen::Dynamic, double>>(m, "Sequence")
        .def("new", &makeModel<PySequence<Eigen::Dynamic, Eigen::Dynamic, double>, py::args>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &PySequence<Eigen::Dynamic, Eigen::Dynamic, double>::forward, "Performs a forward pass through the model.", py::return_value_policy::reference);
        
    py::class_<training::Trainer<Eigen::Dynamic, Eigen::Dynamic, double>>(m, "Trainer")
        .def(py::init<Model<Eigen::Dynamic, Eigen::Dynamic, double> &, std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>())
        .def("train", &training::Trainer<Eigen::Dynamic, Eigen::Dynamic, double>::train, "Trains a model", py::return_value_policy::reference);
}
