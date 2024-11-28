#include "version.h"
#include "include/Model.hpp"
#include "include/Dynamic.hpp"
#include "include/DynamicEncoder.hpp"
#include "include/basic/Layer.hpp"
// #include "include/autoencoder/AutoEncoder.hpp"
#include "include/training/Trainer.hpp"
#include "include/optimizers/Optimizer.hpp"
// #include "include/convolutional/Conv2D.hpp"
// #include "include/convolutional/Pool2D.hpp"
// #include "include/convolutional/UnPool2D.hpp"
// #include "include/convolutional/PoolUnPool2D.hpp"
// #include "include/convolutional/Activation2D.hpp"
// #include "include/convolutional/Reshape.hpp"
// #include "include/convolutional/UNet.hpp"
// #include "include/normalize/ReNorm2D.hpp"
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
using namespace Eigen;

// Scalar datatype and optimizer function used by all installed python models
using PkgScalar = float; 
template<typename T, typename P> using PkgOptimizer = NoOpt<T,P>;

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
            [](const DT& obj){ return obj.getstate(); },
            [](py::tuple data){ return DT(data); }
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
            [](const DT& obj){ return obj.getstate(); },
            [](py::tuple data){ return DT(data); }
            ));
}


PYBIND11_MODULE(neuralnet, m)
{
    m.doc() = "Various neural network implementations";
    m.attr("__version__") = py::cast(PROJECT_VERSION);

    py::class_<DataSource<Vector<PkgScalar, Eigen::Dynamic>, Vector<PkgScalar, Eigen::Dynamic>>, 
                DataSourceTrampoline<Vector<PkgScalar, Eigen::Dynamic>, Vector<PkgScalar, Eigen::Dynamic>>,
                std::shared_ptr<DataSource<Vector<PkgScalar, Eigen::Dynamic>, Vector<PkgScalar, Eigen::Dynamic>>>>(
        m, "DataSource"
    )
        .def(py::init<>())
        .def("size", &DataSource<Vector<PkgScalar, Eigen::Dynamic>, Vector<PkgScalar, Eigen::Dynamic>>::size)
        .def("sample", &DataSource<Vector<PkgScalar, Eigen::Dynamic>, Vector<PkgScalar, Eigen::Dynamic>>::sample);
    py::class_<DataSource<Tensor<PkgScalar,3>, Tensor<PkgScalar,3>>,  
                DataSourceTrampoline<Tensor<PkgScalar,3>, Tensor<PkgScalar,3>>,
                std::shared_ptr<DataSource<Tensor<PkgScalar,3>, Tensor<PkgScalar,3>>>>(
            m, "DataSource2D"
        )
            .def(py::init<>())
            .def("size", &DataSource<Tensor<PkgScalar,3>, Tensor<PkgScalar,3>>::size)
            .def("sample", &DataSource<Tensor<PkgScalar,3>, Tensor<PkgScalar,3>>::sample);

    py::class_<DynamicModel<Vector<PkgScalar, Eigen::Dynamic>>, std::shared_ptr<DynamicModel<Vector<PkgScalar, Eigen::Dynamic>>>, DynamicModelTrampoline<Vector<PkgScalar, Eigen::Dynamic>>>(m, "Model")
        .def(py::init<>());
    py::class_<DynamicEncoder<Vector<PkgScalar, Eigen::Dynamic>, Vector<PkgScalar, Eigen::Dynamic>>, DynamicModel<Vector<PkgScalar, Eigen::Dynamic>>,
        std::shared_ptr<DynamicEncoder<Vector<PkgScalar, Eigen::Dynamic>, Vector<PkgScalar, Eigen::Dynamic>>>, DynamicEncoderTrampoline<Vector<PkgScalar, Eigen::Dynamic>, Vector<PkgScalar, Eigen::Dynamic>>>(m, "Encoder")
        .def(py::init<>());
    py::class_<DynamicModel<Tensor<PkgScalar,3>>, std::shared_ptr<DynamicModel<Tensor<PkgScalar,3>>>, DynamicModelTrampoline<Tensor<PkgScalar,3>>>(m, "Model2D")
        .def(py::init<>());
    py::class_<DynamicEncoder<Tensor<PkgScalar,3>, Tensor<PkgScalar,3>>, DynamicModel<Tensor<PkgScalar,3>>,
        std::shared_ptr<DynamicEncoder<Tensor<PkgScalar,3>, Tensor<PkgScalar,3>>>, DynamicEncoderTrampoline<Tensor<PkgScalar,3>, Tensor<PkgScalar,3>>>(m, "Encoder2D")
        .def(py::init<>());

    make_model<Layer<PkgScalar, ActivationFunc::Linear, PkgOptimizer>, size_t, size_t>(m, "Linear");
    make_model<Layer<PkgScalar, ActivationFunc::ReLU, PkgOptimizer>, size_t, size_t>(m, "ReLU");
    make_model<Layer<PkgScalar, ActivationFunc::Sigmoid, PkgOptimizer>, size_t, size_t>(m, "Sigmoid");
    make_model<Layer<PkgScalar, ActivationFunc::TanH, PkgOptimizer>, size_t, size_t>(m, "TanH");
    make_model<Layer<PkgScalar, ActivationFunc::SoftMax, PkgOptimizer>, size_t, size_t>(m, "SoftMax");

    make_encoder<AutoEncoder<PkgScalar, ActivationFunc::Linear, PkgOptimizer>, size_t, size_t, PkgScalar, PkgScalar>(m, "LinearAutoEncoder");
    make_encoder<AutoEncoder<PkgScalar, ActivationFunc::ReLU, PkgOptimizer>, size_t, size_t, PkgScalar, PkgScalar>(m, "ReLUAutoEncoder");
    make_encoder<AutoEncoder<PkgScalar, ActivationFunc::Sigmoid, PkgOptimizer>, size_t, size_t, PkgScalar, PkgScalar>(m, "SigmoidAutoEncoder");
    make_encoder<AutoEncoder<PkgScalar, ActivationFunc::TanH, PkgOptimizer>, size_t, size_t, PkgScalar, PkgScalar>(m, "TanHAutoEncoder");
    make_encoder<AutoEncoder<PkgScalar, ActivationFunc::SoftMax, PkgOptimizer>, size_t, size_t, PkgScalar, PkgScalar>(m, "SoftMaxAutoEncoder");

    // make_model<Convolution2D<PkgScalar, 5, PkgOptimizer>, Index, Index, PkgScalar, PkgScalar>(m, "Conv2D");
    // make_model<Layer2D<PkgScalar, ActivationFunc::ReLU>>(m, "ReLU2D");
    // make_model<Layer2D<PkgScalar, ActivationFunc::Sigmoid>>(m, "Sigmoid2D");
    // make_model<Layer2D<PkgScalar, ActivationFunc::TanH>>(m, "TanH2D");
    // make_model<Layer2D<PkgScalar, ActivationFunc::SoftMax>>(m, "SoftMax2D");
    // make_model<ReNorm2D<PkgScalar, PkgOptimizer>, int, PkgScalar, PkgScalar, PkgScalar>(m, "BatchRenorm2D");
    // make_model<Pool2D<PkgScalar, 2, PoolMode::Max>>(m, "MaxPool2D");
    // make_model<Pool2D<PkgScalar, 2, PoolMode::Min>>(m, "MinPool2D");
    // make_model<Pool2D<PkgScalar, 2, PoolMode::Mean>>(m, "MeanPool2D");
    // make_model<UnPool2D<PkgScalar, 2, PoolMode::Mean>>(m, "MeanUnPool2D");
    // make_encoder<PoolUnPool2D<PkgScalar, 2, PoolMode::Mean>>(m, "MeanPoolEncoder2D");
    // make_encoder<PoolUnPool2D<PkgScalar, 2, PoolMode::Max>>(m, "MaxPoolEncoder2D");
    // make_encoder<PoolUnPool2D<PkgScalar, 2, PoolMode::Min>>(m, "MinPoolEncoder2D");
    // make_encoder<UNet<PkgScalar, PkgOptimizer>, Index, PkgScalar, PkgScalar, PkgScalar>(m, "UNet")
    //     .def(py::init<Index, PkgScalar, PkgScalar, PkgScalar, bool>());
    // py::class_<Reshape1D<PkgScalar>>(m, "Reshape1D")
    //     .def(py::init<>())
    //     .def("forward", &Reshape1D<PkgScalar>::forward<Tensor<PkgScalar,3>&>, "Performs a forward pass through the model.")
    //     .def("backward", &Reshape1D<PkgScalar>::backward<Vector<PkgScalar,Dynamic>&>, "Performs backpropagation through the model.")
    //     .def("update", &Reshape1D<PkgScalar>::update, "Updates trainable parameters based on current gradient.");

    py::class_<training::Trainer<DynamicModel<Vector<PkgScalar, Eigen::Dynamic>>>>(m, "Trainer")
        .def(py::init<
            DynamicModel<Vector<PkgScalar, Eigen::Dynamic>>&,
            DataSource<DynamicModel<Vector<PkgScalar, Eigen::Dynamic>>::InputType, DynamicModel<Vector<PkgScalar, Eigen::Dynamic>>::OutputType>&,
            Loss<PkgScalar>&
        >())
        .def("train", &training::Trainer<DynamicModel<Vector<PkgScalar, Eigen::Dynamic>>>::train, "Trains a model", py::return_value_policy::automatic);
    
    // py::class_<training::Trainer<DynamicModel<Tensor<PkgScalar,3>>>>(m, "Trainer2D")
    //     .def(py::init<
    //         DynamicModel<Tensor<PkgScalar,3>>&,
    //         DataSource<DynamicModel<Tensor<PkgScalar,3>>::InputType, DynamicModel<Tensor<PkgScalar,3>>::OutputType>&,
    //         Loss<PkgScalar>&
    //     >())
    //     .def("train", &training::Trainer<DynamicModel<Tensor<PkgScalar,3>>>::train, "Trains a model", py::return_value_policy::automatic, 
    //         py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    py::class_<loss::Loss<PkgScalar>, LossTrampoline<PkgScalar>, std::shared_ptr<Loss<PkgScalar>>>(m, "LossBase");
    py::class_<loss::L1<PkgScalar>, loss::Loss<PkgScalar>, std::shared_ptr<L1<PkgScalar>>>(m, "L1Loss").def(py::init<>());
    py::class_<loss::L2<PkgScalar>, loss::Loss<PkgScalar>, std::shared_ptr<L2<PkgScalar>>>(m, "L2Loss").def(py::init<>());
    py::class_<loss::IoU<PkgScalar>, loss::Loss<PkgScalar>, std::shared_ptr<IoU<PkgScalar>>>(m, "IoULoss").def(py::init<>());
    py::class_<loss::BCE<PkgScalar>, loss::Loss<PkgScalar>, std::shared_ptr<BCE<PkgScalar>>>(m, "BCELoss").def(py::init<>());
}
