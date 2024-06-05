#define PYBIND11_DETAILED_ERROR_MESSAGES

#include "basic/Layer.hpp"
#include "basic/Sequence.hpp"
#include "training/Trainer.hpp"
#include "optimizers/Adam.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>

namespace py = pybind11;
using namespace neuralnet;

typedef Model<Eigen::Dynamic, Eigen::Dynamic, double> DynamicBase;

PYBIND11_MODULE(neuralnet, m)
{
    m.doc() = "Various neural network implementations";

    py::class_<DynamicBase>(m, "Model");

    py::class_<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::Linear>, DynamicBase>(m, "Linear")
        .def("new", &makeModel<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::Linear>, size_t, size_t>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::Linear>::forward, "Performs a forward pass through the model.", py::return_value_policy::reference)
        .def("apply_optimizer", &Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::Linear>::apply_optimizer, "Adds an optimization algorithm to the model", py::return_value_policy::reference);

    py::class_<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::ReLU>, DynamicBase>(m, "ReLU")
        .def("new", &makeModel<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::ReLU>, size_t, size_t>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::ReLU>::forward, "Performs a forward pass through the model.", py::return_value_policy::reference)
        .def("apply_optimizer", &Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::ReLU>::apply_optimizer, "Adds an optimization algorithm to the model", py::return_value_policy::reference);

    py::class_<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::Sigmoid>, DynamicBase>(m, "Sigmoid")
        .def("new", &makeModel<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::Sigmoid>, size_t, size_t>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::Sigmoid>::forward, "Performs a forward pass through the model.", py::return_value_policy::reference)
        .def("apply_optimizer", &Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::Sigmoid>::apply_optimizer, "Adds an optimization algorithm to the model", py::return_value_policy::reference);

    py::class_<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::TanH>, DynamicBase>(m, "TanH")
        .def("new", &makeModel<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::TanH>, size_t, size_t>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::TanH>::forward, "Performs a forward pass through the model.", py::return_value_policy::reference)
        .def("apply_optimizer", &Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::TanH>::apply_optimizer, "Adds an optimization algorithm to the model", py::return_value_policy::reference);

    py::class_<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::SoftMax>, DynamicBase>(m, "SoftMax")
        .def("new", &makeModel<Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::SoftMax>, size_t, size_t>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::SoftMax>::forward, "Performs a forward pass through the model.", py::return_value_policy::reference)
        .def("apply_optimizer", &Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::SoftMax>::apply_optimizer, "Adds an optimization algorithm to the model", py::return_value_policy::reference);

    py::class_<Sequence<Eigen::Dynamic, Eigen::Dynamic, double>, DynamicBase>(m, "Sequence")
        .def("new", &makeModel<Sequence<Eigen::Dynamic, Eigen::Dynamic, double>, py::args>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &Sequence<Eigen::Dynamic, Eigen::Dynamic, double>::forward, "Performs a forward pass through the model.", py::return_value_policy::reference)
        .def("apply_optimizer", &Sequence<Eigen::Dynamic, Eigen::Dynamic, double>::apply_optimizer, "Adds an optimization algorithm to the model", py::return_value_policy::reference);

    py::class_<training::Trainer<Eigen::Dynamic, Eigen::Dynamic, double>>(m, "Trainer")
        .def(py::init<DynamicBase &, std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>())
        .def("train", &training::Trainer<Eigen::Dynamic, Eigen::Dynamic, double>::train, "Trains a model", py::return_value_policy::reference);

    py::class_<optimization::Optimizer>(m, "Optimizer");

    py::class_<optimization::Adam, optimization::Optimizer>(m, "AdamOptimizer")
        .def("new", &optimization::makeOptimizer<optimization::Adam, double, double>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference);

    // m.def("getStaticModel", []()
    //       { return Sequence<784, 10, double>(std::vector<Model*>{&Layer<784, 500, double, ActivationFunc::ReLU>(),
    //                                                    &Layer<500, 300, double, ActivationFunc::ReLU>(),
    //                                                    &Layer<300, 300, double, ActivationFunc::ReLU>(),
    //                                                    &Layer<300, 100, double, ActivationFunc::ReLU>(),
    //                                                    &Layer<100, 50, double, ActivationFunc::ReLU>(),
    //                                                    &Layer<50, 10, double, ActivationFunc::SoftMax>()}); }, 
    //                                                    "Statically compiled MNIST classifier.", py::return_value_policy::reference);
}
