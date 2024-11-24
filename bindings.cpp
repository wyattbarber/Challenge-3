#include "version.h"

#include "include/Dynamic.hpp"
#include "include/DynamicEncoder.hpp"
#include "include/basic/Layer.hpp"
#include "include/autoencoder/AutoEncoder.hpp"
#include "include/training/Trainer.hpp"
#include "include/optimizers/Optimizer.hpp"
#include "include/convolutional/Conv2D.hpp"
#include "include/convolutional/Pool2D.hpp"
#include "include/convolutional/UnPool2D.hpp"
#include "include/convolutional/PoolUnPool2D.hpp"
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
    using DT = DynamicBinder<T>;

    return py::class_<DT, DynamicModel<typename T::InputType>, std::shared_ptr<DT>>(m, name)
        .def(py::init<Ts...>())
        .def("forward", 
            static_cast<DT::OutputType (DT::*)(typename DT::InputType&)>(&DT::forward), 
            "Performs a forward pass through the model.", 
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())

        .def("backward", 
            static_cast<DT::InputType (DT::*)(typename DT::OutputType&)>(&DT::backward), 
            "Performs backpropagation through the model.", 
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())

        .def("update", &DT::update, "Updates trainable parameters based on current gradient.", 
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())

        .def(py::pickle(
            [](const DT& obj){ return T::getstate(*obj.getmodel()); },
            [](py::tuple data){ return DT(T::setstate(data)); }
            ));
}

template<class T, typename... Ts>
auto make_encoder(py::module m, const char* name)
{
    using DT = DynamicEncoderBinder<T>;

    return py::class_<DT, DynamicEncoder<typename T::InputType, typename T::LatentType>, std::shared_ptr<DT>>(m, name)
        .def(py::init<Ts...>())
        .def("forward", 
            static_cast<DT::OutputType (DT::*)(typename DT::InputType&)>(&DT::forward), 
            "Performs a forward pass through the model.", 
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())

        .def("backward", 
            static_cast<DT::InputType (DT::*)(typename DT::OutputType&)>(&DT::backward), 
            "Performs backpropagation through the model.", 
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())

        .def("encode", 
            static_cast<DT::LatentType (DT::*)(typename DT::InputType&)>(&DT::encode), 
            "Generates a latent embedding", 
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())

        .def("decode", 
            static_cast<DT::OutputType (DT::*)(typename DT::LatentType&)>(&DT::decode), 
            "Reconstructs a latent embedding", 
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())

        .def("backward_encode", 
            static_cast<DT::InputType (DT::*)(typename DT::LatentType&)>(&DT::backward_encode), 
            "Backpropagates error for the encoding portion", 
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())

        .def("backward_decode", 
            static_cast<DT::LatentType (DT::*)(typename DT::OutputType&)>(&DT::backward_decode), 
            "Backpropagates error for the decoding portion", 
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())

        .def("update", &DT::update, "Updates trainable parameters based on current gradient.", 
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())

        .def(py::pickle(
            [](const DT& obj){ return T::getstate(*obj.getmodel()); },
            [](py::tuple data){ return DT(T::setstate(data)); }
            ));
}


PYBIND11_MODULE(neuralnet, m)
{
    m.doc() = "Various neural network implementations";
    m.attr("__version__") = py::cast(PROJECT_VERSION);

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

    py::class_<DynamicModel<Eigen::VectorXd>, std::shared_ptr<DynamicModel<Eigen::VectorXd>>, DynamicModelTrampoline<Eigen::VectorXd>>(m, "Model")
        .def(py::init<>());
    py::class_<DynamicEncoder<Eigen::VectorXd, Eigen::VectorXd>, DynamicModel<Eigen::VectorXd>,
        std::shared_ptr<DynamicEncoder<Eigen::VectorXd, Eigen::VectorXd>>, DynamicEncoderTrampoline<Eigen::VectorXd, Eigen::VectorXd>>(m, "Encoder")
        .def(py::init<>());
    py::class_<DynamicModel<Eigen::Tensor<double,3>>, std::shared_ptr<DynamicModel<Eigen::Tensor<double,3>>>, DynamicModelTrampoline<Eigen::Tensor<double,3>>>(m, "Model2D")
        .def(py::init<>());
    py::class_<DynamicEncoder<Eigen::Tensor<double,3>, Eigen::Tensor<double,3>>, DynamicModel<Eigen::Tensor<double,3>>,
        std::shared_ptr<DynamicEncoder<Eigen::Tensor<double,3>, Eigen::Tensor<double,3>>>, DynamicEncoderTrampoline<Eigen::Tensor<double,3>, Eigen::Tensor<double,3>>>(m, "Encoder2D")
        .def(py::init<>());

    make_model<Layer<double, ActivationFunc::Linear, OptimizerClass::Adam>, size_t, size_t, double, double>(m, "Linear");
    make_model<Layer<double, ActivationFunc::ReLU, OptimizerClass::Adam>, size_t, size_t, double, double>(m, "ReLU");
    make_model<Layer<double, ActivationFunc::Sigmoid, OptimizerClass::Adam>, size_t, size_t, double, double>(m, "Sigmoid");
    make_model<Layer<double, ActivationFunc::TanH, OptimizerClass::Adam>, size_t, size_t, double, double>(m, "TanH");
    make_model<Layer<double, ActivationFunc::SoftMax, OptimizerClass::Adam>, size_t, size_t, double, double>(m, "SoftMax");

    make_encoder<AutoEncoder<double, ActivationFunc::Linear, OptimizerClass::Adam>, size_t, size_t, double, double>(m, "LinearAutoEncoder");
    make_encoder<AutoEncoder<double, ActivationFunc::ReLU, OptimizerClass::Adam>, size_t, size_t, double, double>(m, "ReLUAutoEncoder");
    make_encoder<AutoEncoder<double, ActivationFunc::Sigmoid, OptimizerClass::Adam>, size_t, size_t, double, double>(m, "SigmoidAutoEncoder");
    make_encoder<AutoEncoder<double, ActivationFunc::TanH, OptimizerClass::Adam>, size_t, size_t, double, double>(m, "TanHAutoEncoder");
    make_encoder<AutoEncoder<double, ActivationFunc::SoftMax, OptimizerClass::Adam>, size_t, size_t, double, double>(m, "SoftMaxAutoEncoder");

    make_model<Convolution2D<double, 5, OptimizerClass::Adam>, Eigen::Index, Eigen::Index, double, double>(m, "Conv2D");
    make_model<Layer2D<double, ActivationFunc::ReLU>>(m, "ReLU2D");
    make_model<Layer2D<double, ActivationFunc::Sigmoid>>(m, "Sigmoid2D");
    make_model<Layer2D<double, ActivationFunc::TanH>>(m, "TanH2D");
    make_model<Layer2D<double, ActivationFunc::SoftMax>>(m, "SoftMax2D");
    make_model<Pool2D<double, 2, PoolMode::Max>>(m, "MaxPool2D");
    make_model<Pool2D<double, 2, PoolMode::Min>>(m, "MinPool2D");
    make_model<Pool2D<double, 2, PoolMode::Mean>>(m, "MeanPool2D");
    make_model<UnPool2D<double, 2, PoolMode::Mean>>(m, "MeanUnPool2D");
    make_encoder<PoolUnPool2D<double, 2, PoolMode::Mean>>(m, "MeanPoolEncoder2D");
    make_encoder<PoolUnPool2D<double, 2, PoolMode::Max>>(m, "MaxPoolEncoder2D");
    make_encoder<PoolUnPool2D<double, 2, PoolMode::Min>>(m, "MinPoolEncoder2D");
    py::class_<Reshape1D<double>>(m, "Reshape1D")
        .def(py::init<>())
        .def("forward", &Reshape1D<double>::forward<Eigen::Tensor<double,3>&>, "Performs a forward pass through the model.")
        .def("backward", &Reshape1D<double>::backward<Eigen::Vector<double,Eigen::Dynamic>&>, "Performs backpropagation through the model.")
        .def("update", &Reshape1D<double>::update, "Updates trainable parameters based on current gradient.");

    py::class_<training::Trainer<DynamicModel<Eigen::VectorXd>>>(m, "Trainer")
        .def(py::init<
            DynamicModel<Eigen::VectorXd>&,
            DataSource<DynamicModel<Eigen::VectorXd>::InputType, DynamicModel<Eigen::VectorXd>::OutputType>&,
            Loss<double>&
        >())
        .def("train", &training::Trainer<DynamicModel<Eigen::VectorXd>>::train, "Trains a model", py::return_value_policy::automatic);
    
    py::class_<training::Trainer<DynamicModel<Eigen::Tensor<double,3>>>>(m, "Trainer2D")
        .def(py::init<
            DynamicModel<Eigen::Tensor<double,3>>&,
            DataSource<DynamicModel<Eigen::Tensor<double,3>>::InputType, DynamicModel<Eigen::Tensor<double,3>>::OutputType>&,
            Loss<double>&
        >())
        .def("train", &training::Trainer<DynamicModel<Eigen::Tensor<double,3>>>::train, "Trains a model", py::return_value_policy::automatic, 
            py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    py::class_<loss::Loss<double>, LossTrampoline<double>, std::shared_ptr<Loss<double>>>(m, "LossBase");
    py::class_<loss::L1<double>, loss::Loss<double>, std::shared_ptr<L1<double>>>(m, "L1Loss").def(py::init<>());
    py::class_<loss::L2<double>, loss::Loss<double>, std::shared_ptr<L2<double>>>(m, "L2Loss").def(py::init<>());
    py::class_<loss::IoU<double>, loss::Loss<double>, std::shared_ptr<IoU<double>>>(m, "IoULoss").def(py::init<>());
    py::class_<loss::BCE<double>, loss::Loss<double>, std::shared_ptr<BCE<double>>>(m, "BCELoss").def(py::init<>());
}
