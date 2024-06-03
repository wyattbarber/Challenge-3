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


PYBIND11_MODULE(neuralnet, m)
{
    m.doc() = "Various neural network implementations";

    py::class_<Model>(m, "Model");
    
    py::class_<Layer<ActivationFunc::Linear>, Model>(m, "Linear")
        .def("new", &makeModel<Layer<ActivationFunc::Linear>, size_t, size_t>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &Layer<ActivationFunc::Linear>::forward, "Performs a forward pass through the model.", py::return_value_policy::reference)
        .def("apply_optimizer", &Layer<ActivationFunc::Linear>::apply_optimizer, "Adds an optimization algorithm to the model", py::return_value_policy::reference);

    py::class_<Layer<ActivationFunc::ReLU>, Model>(m, "ReLU")
        .def("new", &makeModel<Layer<ActivationFunc::ReLU>, size_t, size_t>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &Layer<ActivationFunc::ReLU>::forward, "Performs a forward pass through the model.", py::return_value_policy::reference)
        .def("apply_optimizer", &Layer<ActivationFunc::ReLU>::apply_optimizer, "Adds an optimization algorithm to the model", py::return_value_policy::reference);
    
    py::class_<Layer<ActivationFunc::Sigmoid>, Model>(m, "Sigmoid")
        .def("new", &makeModel<Layer<ActivationFunc::Sigmoid>, size_t, size_t>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &Layer<ActivationFunc::Sigmoid>::forward, "Performs a forward pass through the model.", py::return_value_policy::reference)
        .def("apply_optimizer", &Layer<ActivationFunc::Sigmoid>::apply_optimizer, "Adds an optimization algorithm to the model", py::return_value_policy::reference);

    py::class_<Layer<ActivationFunc::TanH>, Model>(m, "TanH")
        .def("new", &makeModel<Layer<ActivationFunc::TanH>, size_t, size_t>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &Layer<ActivationFunc::TanH>::forward, "Performs a forward pass through the model.", py::return_value_policy::reference)
        .def("apply_optimizer", &Layer<ActivationFunc::TanH>::apply_optimizer, "Adds an optimization algorithm to the model", py::return_value_policy::reference);
    
    py::class_<Layer<ActivationFunc::SoftMax>, Model>(m, "SoftMax")
        .def("new", &makeModel<Layer<ActivationFunc::SoftMax>, size_t, size_t>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &Layer<ActivationFunc::SoftMax>::forward, "Performs a forward pass through the model.", py::return_value_policy::reference)
        .def("apply_optimizer", &Layer<ActivationFunc::SoftMax>::apply_optimizer, "Adds an optimization algorithm to the model", py::return_value_policy::reference);

    py::class_<Sequence, Model>(m, "Sequence")
        .def("new", &makeModel<Sequence, py::args>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &Sequence::forward, "Performs a forward pass through the model.", py::return_value_policy::reference)
        .def("apply_optimizer", &Sequence::apply_optimizer, "Adds an optimization algorithm to the model", py::return_value_policy::reference);


    py::class_<training::Trainer>(m, "Trainer")
        .def(py::init<Model&, std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>())
        .def("train", &training::Trainer::train, "Trains a model", py::return_value_policy::reference);

    
    py::class_<optimization::Optimizer>(m, "Optimizer");
    
    py::class_<optimization::Adam, optimization::Optimizer>(m, "AdamOptimizer")
        .def("new", &optimization::makeOptimizer<optimization::Adam, double, double>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference);
}
