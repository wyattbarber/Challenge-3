#define PYBIND11_DETAILED_ERROR_MESSAGES

#include "basic/Layer.hpp"
#include "basic/Sequence.hpp"
#include "training/Trainer.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>

namespace py = pybind11;
using namespace neuralnet;

template<class ModelType, typename... CTypes>
constexpr void bind_model(py::module_& m, const char* pyname)
{
    py::class_<ModelType, Model>(m, pyname)
        .def("new", &makeModel<ModelType, CTypes...>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &ModelType::forward, "Performs a forward pass through the model.", py::return_value_policy::reference); 
}

template<class TrainerType>
constexpr void bind_trainer(py::module_& m, const char* pyname)
{
    py::class_<TrainerType>(m, pyname)
        .def(py::init<Model&, std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>())
        .def("train", &TrainerType::train, "Trains a model", py::return_value_policy::reference); 
}

PYBIND11_MODULE(neuralnet, m)
{
    m.doc() = "Various neural network implementations";

    py::class_<Model>(m, "Model");

    bind_model<Layer<ActivationFunc::Linear>, size_t, size_t>(m, "Linear");
    bind_model<Layer<ActivationFunc::ReLU>, size_t, size_t>(m, "ReLU");
    bind_model<Layer<ActivationFunc::Sigmoid>, size_t, size_t>(m, "Sigmoid");
    bind_model<Layer<ActivationFunc::TanH>, size_t, size_t>(m, "TanH");
    bind_model<Layer<ActivationFunc::SoftMax>, size_t, size_t>(m, "SoftMax");
    bind_model<Sequence, py::args>(m, "Sequence");

    bind_trainer<training::Trainer>(m, "Trainer");
}
