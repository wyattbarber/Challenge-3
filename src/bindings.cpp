#define EIGEN_STACK_ALLOCATION_LIMIT (128000 * 2)

#include "basic/Layer.hpp"
#include "basic/PySequence.hpp"
#include "basic/Compound.hpp"
#include "basic/Converter.hpp"
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

class TestModel : public Model<784, 10, double>
{
public:
    TestModel(double b1, double b2) : l1(784, 500, b1, b2),
                                      l2(500, 300, b1, b2),
                                      l3(300, 300, b1, b2),
                                      l4(b1, b2),
                                      l5(b1, b2),
                                      l6(b1, b2)
    {
    }

    Eigen::Vector<double, 10> forward(Eigen::Vector<double, 784> &input)
    {
        return sequential::forward(input, c1, l1, l2, l3, c2, l4, l5, l6);
    }

    Eigen::Vector<double, 784> backward(Eigen::Vector<double, 10> &error)
    {
        return sequential::backward(error, c1, l1, l2, l3, c2, l4, l5, l6);
    }

    void update(double rate)
    {
        sequential::update(rate, l1, l2, l3, l4, l5, l6);
    }

protected:
    conversions::Static2Dynamic<784, double> c1;
    Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::ReLU, OptimizerClass::Adam> l1;
    Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::ReLU, OptimizerClass::Adam> l2;
    Layer<Eigen::Dynamic, Eigen::Dynamic, double, ActivationFunc::ReLU, OptimizerClass::Adam> l3;
    conversions::Dynamic2Static<300, double> c2;
    Layer<300, 100, double, ActivationFunc::ReLU, OptimizerClass::Adam> l4;
    Layer<100, 50, double, ActivationFunc::ReLU, OptimizerClass::Adam> l5;
    Layer<50, 10, double, ActivationFunc::SoftMax, OptimizerClass::Adam> l6;
};

TestModel s(0.9, 0.999);

PYBIND11_MODULE(neuralnet, m)
{
    m.doc() = "Various neural network implementations";

    py::class_<Model<Eigen::Dynamic, Eigen::Dynamic, double>>(m, "Model");

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

    py::class_<PySequence<double>, Model<Eigen::Dynamic, Eigen::Dynamic, double>>(m, "Sequence")
        .def("new", &makeModel<PySequence<double>, py::args>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &PySequence<double>::forward, "Performs a forward pass through the model.", py::return_value_policy::reference);

    py::class_<training::Trainer<Eigen::Dynamic, Eigen::Dynamic, double>>(m, "Trainer")
        .def(py::init<Model<Eigen::Dynamic, Eigen::Dynamic, double> &, std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>())
        .def("train", &training::Trainer<Eigen::Dynamic, Eigen::Dynamic, double>::train, "Trains a model", py::return_value_policy::reference);

    py::class_<Model<784, 10, double>>(m, "_TestBase");

    py::class_<TestModel, Model<784, 10, double>>(m, "TestModel")
        .def("new", &makeModel<TestModel, double, double>, "Get a pointer to a new instance of this class.", py::return_value_policy::reference)
        .def("forward", &TestModel::forward, "Performs a forward pass through the model.", py::return_value_policy::reference);

    py::class_<training::Trainer<784, 10, double>>(m, "StaticTrainer")
        .def(py::init<Model<784, 10, double> &, std::vector<Eigen::Vector<double, 784>>, std::vector<Eigen::Vector<double, 10>>>())
        .def("train", &training::Trainer<784, 10, double>::train, "Trains a model", py::return_value_policy::reference);
}
