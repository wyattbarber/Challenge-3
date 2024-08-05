#define EIGEN_STACK_ALLOCATION_LIMIT (128000 * 2)

#include "basic/Layer.hpp"
#include "basic/PySequence.hpp"
#include "basic/Compound.hpp"
#include "autoencoder/AutoEncoder.hpp"
#include "training/Trainer.hpp"
#include "optimizers/Optimizer.hpp"
#include <tuple>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>

namespace py = pybind11;
using namespace neuralnet;
using namespace optimization;

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::VectorXd>);

class TestModel : public Model<TestModel>
{
public:
    TestModel(double b1, double b2) : l1(784, 500, b1, b2),
                                      l2(500, 300, b1, b2),
                                      l3(300, 100, b1, b2),
                                      l4(100, 100, b1, b2),
                                      l5(100, 50, b1, b2),
                                      l6(50, 10, b1, b2)
    {
    }

    Eigen::VectorXd forward(Eigen::VectorXd &input)
    {
        return sequential::forward(input, l1, l2, l3, l4, l5, l6);
    }

    Eigen::VectorXd backward(Eigen::VectorXd &error)
    {
        return sequential::backward(error, l1, l2, l3, l4, l5, l6);
    }

    void update(double rate)
    {
        sequential::update(rate, l1, l2, l3, l4, l5, l6);
    }

protected:
    Layer<double, ActivationFunc::TanH, OptimizerClass::Adam> l1;
    Layer<double, ActivationFunc::ReLU, OptimizerClass::Adam> l2;
    Layer<double, ActivationFunc::ReLU, OptimizerClass::Adam> l3;
    Layer<double, ActivationFunc::ReLU, OptimizerClass::Adam> l4;
    Layer<double, ActivationFunc::ReLU, OptimizerClass::Adam> l5;
    Layer<double, ActivationFunc::SoftMax, OptimizerClass::Adam> l6;
};


PYBIND11_MODULE(neuralnet, m)
{
    m.doc() = "Various neural network implementations";

    py::class_<DynamicModel<double>>(m, "Model");

    #define Linear_CLASS DynamicBinder<double, Layer<double, ActivationFunc::Linear, OptimizerClass::Adam>>
    py::class_<Linear_CLASS, DynamicModel<double>>(m, "Linear")
        .def("new", &makeModel<Linear_CLASS, size_t, size_t, double, double>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &Linear_CLASS::forward, "Performs a forward pass through the model.", py::return_value_policy::reference);
    #undef Linear_CLASS

    #define ReLU_CLASS DynamicBinder<double, Layer<double, ActivationFunc::ReLU, OptimizerClass::Adam>>
    py::class_<ReLU_CLASS, DynamicModel<double>>(m, "ReLU")
        .def("new", &makeModel<ReLU_CLASS, size_t, size_t, double, double>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &ReLU_CLASS::forward, "Performs a forward pass through the model.", py::return_value_policy::reference);
    #undef ReLU_CLASS

    #define Sigmoid_CLASS DynamicBinder<double, Layer<double, ActivationFunc::Sigmoid, OptimizerClass::Adam>>
    py::class_<Sigmoid_CLASS, DynamicModel<double>>(m, "Sigmoid")
        .def("new", &makeModel<Sigmoid_CLASS, size_t, size_t, double, double>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &Sigmoid_CLASS::forward, "Performs a forward pass through the model.", py::return_value_policy::reference);
    #undef Sigmoid_CLASS

    #define TanH_CLASS DynamicBinder<double, Layer<double, ActivationFunc::TanH, OptimizerClass::Adam>>
    py::class_<TanH_CLASS, DynamicModel<double>>(m, "TanH")
        .def("new", &makeModel<TanH_CLASS, size_t, size_t, double, double>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &TanH_CLASS::forward, "Performs a forward pass through the model.", py::return_value_policy::reference);
    #undef TanH_CLASS

    #define SoftMax_CLASS DynamicBinder<double, Layer<double, ActivationFunc::SoftMax, OptimizerClass::Adam>>
    py::class_<SoftMax_CLASS, DynamicModel<double>>(m, "SoftMax")
        .def("new", &makeModel<SoftMax_CLASS, size_t, size_t, double, double>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &SoftMax_CLASS::forward, "Performs a forward pass through the model.", py::return_value_policy::reference);
    #undef SoftMax_CLASS

    #define AutoEncoder_CLASS DynamicBinder<double, AutoEncoder<double, ActivationFunc::ReLU, OptimizerClass::Adam>>
    py::class_<AutoEncoder_CLASS, DynamicModel<double>>(m, "AutoEncoder")
        .def("new", &makeModel<AutoEncoder_CLASS, size_t, size_t, double, double>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &AutoEncoder_CLASS::forward, "Performs a forward pass through the entire model.", py::return_value_policy::reference);
    #undef AutoEncoder_CLASS

    py::class_<DynamicBinder<double, PySequence<double>>, DynamicModel<double>>(m, "Sequence")
        .def("new", &makeModel<DynamicBinder<double, PySequence<double>>, py::args>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &DynamicBinder<double, PySequence<double>>::forward, "Performs a forward pass through the model.", py::return_value_policy::reference);

    py::class_<training::Trainer<DynamicBinder<double, PySequence<double>>>>(m, "Trainer")
        .def(py::init<DynamicBinder<double, PySequence<double>>&, std::vector<Eigen::VectorXd>&, std::vector<Eigen::VectorXd>&>())
        .def("train", &training::Trainer<DynamicBinder<double, PySequence<double>>>::train, "Trains a model", py::return_value_policy::reference);


    py::class_<TestModel>(m, "TestModel")
        .def("new", &makeModel<TestModel, double, double>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &TestModel::forward, "Performs a forward pass through the model.", py::return_value_policy::reference);

    py::class_<training::Trainer<TestModel>>(m, "StaticTrainer")
        .def(py::init<TestModel&, std::vector<Eigen::VectorXd>&, std::vector<Eigen::VectorXd>&>())
        .def("train", &training::Trainer<TestModel>::train, "Trains a model", py::return_value_policy::reference);
}
