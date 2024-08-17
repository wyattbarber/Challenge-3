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

// PYBIND11_MAKE_OPAQUE(std::vector<Eigen::VectorXd>);

class TestModel : public Model<TestModel>
{
protected:
    Layer<double, ActivationFunc::TanH, OptimizerClass::Adam> l1;
    Layer<double, ActivationFunc::ReLU, OptimizerClass::Adam> l2;
    Layer<double, ActivationFunc::ReLU, OptimizerClass::Adam> l3;
    Layer<double, ActivationFunc::ReLU, OptimizerClass::Adam> l4;
    Layer<double, ActivationFunc::ReLU, OptimizerClass::Adam> l5;
    Layer<double, ActivationFunc::SoftMax, OptimizerClass::Adam> l6;

public:
    typedef decltype(l1)::InputType InputType; 
    typedef decltype(l6)::OutputType OutputType; 

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
};


template<class T, typename... Ts>
void make_model(py::module m, const char* name)
{
    py::class_<T, DynamicModel<double>, std::shared_ptr<T>>(m, name)
    .def(py::init<Ts...>())
    .def("forward", &T::forward, "Performs a forward pass through the model.");
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

    make_model<DynamicBinder<double, AutoEncoder<double, ActivationFunc::Linear, OptimizerClass::Adam>>, size_t, size_t, double, double>(m, "AutoEncoder");

    make_model<DynamicBinder<double, PySequence<double>>, std::vector<DynamicModel<double>*>>(m, "Sequence");

    py::class_<training::Trainer<DynamicModel<double>>>(m, "Trainer")
        .def(py::init<
            std::shared_ptr<DynamicModel<double>>,
            std::vector<DynamicModel<double>::InputType>,
            std::vector<DynamicModel<double>::OutputType>
        >())
        .def("train", &training::Trainer<DynamicModel<double>>::train, "Trains a model", py::return_value_policy::automatic);

    make_model<DynamicBinder<double, TestModel>, double, double>(m, "TestModel");

    m.def("fixed_trainer", [](std::vector<TestModel::InputType> inputs, std::vector<TestModel::OutputType> outputs, size_t N, double a){
        py::print("Creating model");        
        auto model = std::make_shared<TestModel>(0.9, 0.999);
        py::print("Creating trainer");
        auto trainer = training::Trainer<TestModel>(model, inputs, outputs);
        py::print("Starting training");
        return trainer.train(N, a);
    });
}
