#include "include/basic/Layer.hpp"
#include "include/basic/PySequence.hpp"
#include "include/basic/Compound.hpp"
#include "include/autoencoder/AutoEncoder.hpp"
#include "include/autoencoder/DeepAutoEncoder.hpp"
#include "include/training/Trainer.hpp"
#include "include/optimizers/Optimizer.hpp"
#include "include/convolutional/Conv2D.hpp"
#include "include/convolutional/Pool2D.hpp"
#include "include/convolutional/UnPool2D.hpp"
#include "include/convolutional/Activation2D.hpp"
#include "include/convolutional/Reshape.hpp"
#include "include/datasource/DataSource.hpp"
#include "include/loss/Loss.hpp"
#include "include/loss/L1.hpp"
#include "include/loss/L2.hpp"
#include "include/loss/IoU.hpp"
#include "include/loss/BCE.hpp"
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
using namespace datasource;
using namespace loss;

template<class T, typename... Ts>
auto make_model(py::module m, const char* name)
{
    if constexpr (std::is_convertible_v<typename T::OutputType, Eigen::Tensor<double, 3>>)
    {
        return py::class_<T, DynamicTensor3Model<double>, std::shared_ptr<T>>(m, name)
        .def(py::init<Ts...>())
        .def("forward", static_cast<T::OutputType (T::*)(typename T::InputType&)>(&T::forward), "Performs a forward pass through the model.", 
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("backward", static_cast<T::InputType (T::*)(typename T::OutputType&)>(&T::backward), "Performs backpropagation through the model.", 
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("update", &T::update, "Updates trainable parameters based on current gradient.", 
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    }
    else
    {
        return py::class_<T, DynamicModel<double>, std::shared_ptr<T>>(m, name)
        .def(py::init<Ts...>())
        .def("forward", static_cast<T::OutputType (T::*)(typename T::InputType&)>(&T::forward), "Performs a forward pass through the model.", 
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("backward", static_cast<T::InputType (T::*)(typename T::OutputType&)>(&T::backward), "Performs backpropagation through the model.", 
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("update", &T::update, "Updates trainable parameters based on current gradient.", 
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    }
}


PYBIND11_MODULE(neuralnet, m)
{
    m.doc() = "Various neural network implementations";

    py::class_<DataSource<Eigen::VectorXd, Eigen::VectorXd>, 
                DataSourceTrampoline<Eigen::VectorXd, Eigen::VectorXd>,
                std::shared_ptr<DataSource<Eigen::VectorXd, Eigen::VectorXd>>>(
        m, "DataSource"
    )
        .def(py::init<>())
        .def("size", &DataSource<Eigen::VectorXd, Eigen::VectorXd>::size)
        .def("sample", &DataSource<Eigen::VectorXd, Eigen::VectorXd>::sample);
    py::class_<DataSource<Eigen::Tensor<double,3>, Eigen::Tensor<double,3>>,  
                DataSourceTrampoline<Eigen::Tensor<double,3>, Eigen::Tensor<double,3>>,
                std::shared_ptr<DataSource<Eigen::Tensor<double,3>, Eigen::Tensor<double,3>>>>(
            m, "DataSource2D"
        )
            .def(py::init<>())
            .def("size", &DataSource<Eigen::Tensor<double,3>, Eigen::Tensor<double,3>>::size)
            .def("sample", &DataSource<Eigen::Tensor<double,3>, Eigen::Tensor<double,3>>::sample);

    py::class_<DynamicModel<double>, std::shared_ptr<DynamicModel<double>>, DynamicModelTrampoline<double>>(m, "Model")
        .def(py::init<>());
    py::class_<DynamicTensor3Model<double>, std::shared_ptr<DynamicTensor3Model<double>>, DynamicTensor3ModelTrampoline<double>>(m, "Model2D")
        .def(py::init<>());

    make_model<DynamicBinder<double, Layer<double, ActivationFunc::Linear, OptimizerClass::Adam>>, size_t, size_t, double, double>(m, "Linear");
    make_model<DynamicBinder<double, Layer<double, ActivationFunc::ReLU, OptimizerClass::Adam>>, size_t, size_t, double, double>(m, "ReLU");
    make_model<DynamicBinder<double, Layer<double, ActivationFunc::Sigmoid, OptimizerClass::Adam>>, size_t, size_t, double, double>(m, "Sigmoid");
    make_model<DynamicBinder<double, Layer<double, ActivationFunc::TanH, OptimizerClass::Adam>>, size_t, size_t, double, double>(m, "TanH");
    make_model<DynamicBinder<double, Layer<double, ActivationFunc::SoftMax, OptimizerClass::Adam>>, size_t, size_t, double, double>(m, "SoftMax");

    make_model<DynamicBinder<double, AutoEncoder<double, ActivationFunc::Linear, OptimizerClass::Adam>>, size_t, size_t, double, double>(m, "LinearAutoEncoder");
    make_model<DynamicBinder<double, AutoEncoder<double, ActivationFunc::ReLU, OptimizerClass::Adam>>, size_t, size_t, double, double>(m, "ReLUAutoEncoder");
    make_model<DynamicBinder<double, AutoEncoder<double, ActivationFunc::Sigmoid, OptimizerClass::Adam>>, size_t, size_t, double, double>(m, "SigmoidAutoEncoder");
    make_model<DynamicBinder<double, AutoEncoder<double, ActivationFunc::TanH, OptimizerClass::Adam>>, size_t, size_t, double, double>(m, "TanHAutoEncoder");
    make_model<DynamicBinder<double, AutoEncoder<double, ActivationFunc::SoftMax, OptimizerClass::Adam>>, size_t, size_t, double, double>(m, "SoftMaxAutoEncoder");

    make_model<DynamicBinder<double, PySequence<DynamicModel<double>>>, std::vector<std::shared_ptr<DynamicModel<double>>>>(m, "Sequence");
    make_model<DynamicTensor3Binder<double, PySequence<DynamicTensor3Model<double>>>, std::vector<std::shared_ptr<DynamicTensor3Model<double>>>>(m, "Sequence2D");

    make_model<DynamicTensor3Binder<double, Convolution2D<double, 5, OptimizerClass::Adam>>, Eigen::Index, Eigen::Index, double, double>(m, "Conv2D");
    make_model<DynamicTensor3Binder<double, Layer2D<double, ActivationFunc::ReLU>>>(m, "ReLU2D");
    make_model<DynamicTensor3Binder<double, Layer2D<double, ActivationFunc::Sigmoid>>>(m, "Sigmoid2D");
    make_model<DynamicTensor3Binder<double, Layer2D<double, ActivationFunc::TanH>>>(m, "TanH2D");
    make_model<DynamicTensor3Binder<double, Layer2D<double, ActivationFunc::SoftMax>>>(m, "SoftMax2D");
    py::class_<Reshape1D<double>>(m, "Reshape1D")
        .def(py::init<>())
        .def("forward", &Reshape1D<double>::forward<Eigen::Tensor<double,3>&>, "Performs a forward pass through the model.")
        .def("backward", &Reshape1D<double>::backward<Eigen::Vector<double,Eigen::Dynamic>&>, "Performs backpropagation through the model.")
        .def("update", &Reshape1D<double>::update, "Updates trainable parameters based on current gradient.");
    py::class_<Pool2D<double, 2, PoolMode::Max>>(m, "MaxPool2D")
        .def(py::init<>())
        .def("forward", &Pool2D<double, 2, PoolMode::Max>::forward<Eigen::Tensor<double,3>&>, "Performs a forward pass through the model.")
        .def("backward", &Pool2D<double, 2, PoolMode::Max>::backward<Eigen::Tensor<double,3>&>, "Performs backpropagation through the model.")
        .def("update", &Pool2D<double, 2, PoolMode::Max>::update, "Updates trainable parameters based on current gradient.");
    py::class_<UnPool2D<double, 2, PoolMode::Max>>(m, "MaxUnPool2D")
        .def(py::init<Pool2D<double, 2, PoolMode::Max>&>())
        .def("forward", &UnPool2D<double, 2, PoolMode::Max>::forward<Eigen::Tensor<double,3>&>, "Performs a forward pass through the model.")
        .def("backward", &UnPool2D<double, 2, PoolMode::Max>::backward<Eigen::Tensor<double,3>&>, "Performs backpropagation through the model.")
        .def("update", &UnPool2D<double, 2, PoolMode::Max>::update, "Updates trainable parameters based on current gradient.");

    py::class_<training::Trainer<DynamicModel<double>>>(m, "Trainer")
        .def(py::init<
            DynamicModel<double>&,
            DataSource<DynamicModel<double>::InputType, DynamicModel<double>::OutputType>&,
            Loss<double>&
        >())
        .def("train", &training::Trainer<DynamicModel<double>>::train, "Trains a model", py::return_value_policy::automatic);
    
    py::class_<training::Trainer<DynamicTensor3Model<double>>>(m, "Trainer2D")
        .def(py::init<
            DynamicTensor3Model<double>&,
            DataSource<DynamicTensor3Model<double>::InputType, DynamicTensor3Model<double>::OutputType>&,
            Loss<double>&
        >())
        .def("train", &training::Trainer<DynamicTensor3Model<double>>::train, "Trains a model", py::return_value_policy::automatic, 
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    py::class_<loss::Loss<double>, LossTrampoline<double>, std::shared_ptr<Loss<double>>>(m, "LossBase");
    py::class_<loss::L1<double>, loss::Loss<double>, std::shared_ptr<L1<double>>>(m, "L1Loss").def(py::init<>());
    py::class_<loss::L2<double>, loss::Loss<double>, std::shared_ptr<L2<double>>>(m, "L2Loss").def(py::init<>());
    py::class_<loss::IoU<double>, loss::Loss<double>, std::shared_ptr<IoU<double>>>(m, "IoULoss").def(py::init<>());
    py::class_<loss::BCE<double>, loss::Loss<double>, std::shared_ptr<BCE<double>>>(m, "BCELoss").def(py::init<>());
}
