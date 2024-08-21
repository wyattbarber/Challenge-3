#ifndef _AUTOENCODER_HPP
#define _AUTOENCODER_HPP

#include "../Model.hpp"
#include "../basic/Activation.hpp"
#include "../optimizers/Optimizer.hpp"

using namespace optimization;

namespace neuralnet {

    template <typename T, ActivationFunc F, OptimizerClass C>
    class AutoEncoder : public Model<AutoEncoder<T, F, C>>
    {
    public:
        typedef Eigen::Vector<T, Eigen::Dynamic> InputType;
        typedef InputType OutputType;
        typedef Eigen::Vector<T, Eigen::Dynamic> LatentType;

        template <typename... Ts>
        AutoEncoder(Ts... Args)
        {            
            auto args = std::tuple<Ts...>(Args...);

            this->in_size = std::get<0>(args);
            this->latent_size = std::get<1>(args);

            // Apply he initialization
            this->W = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Random(in_size, latent_size).unaryExpr([in_size = in_size](double x)
                                                                                        { return x * std::sqrt(2.0 / static_cast<double>(in_size)); });

            this->blt = LatentType::Zero(latent_size);
            this->alt = LatentType::Zero(latent_size);
            this->dlt = LatentType::Zero(latent_size);
            this->zlt  = InputType::Zero(in_size);

            this->brc = InputType::Zero(in_size);
            this->arc = InputType::Zero(in_size);
            this->drc = InputType::Zero(in_size);
            this->zrc = LatentType::Zero(latent_size);

            this->in = InputType::Zero(in_size);

            if constexpr (C == OptimizerClass::Adam)
            {
                adam_w.m = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(in_size, latent_size);
                adam_w.v = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(in_size, latent_size);
                adam_brc.m = InputType::Zero(in_size);
                adam_brc.v = InputType::Zero(in_size);                
                adam_blt.m = LatentType::Zero(latent_size);
                adam_blt.v = LatentType::Zero(latent_size);

                adam_w.b1 = std::get<2>(args);
                adam_w.b2 = std::get<3>(args);
                adam_brc.b1 = std::get<2>(args);
                adam_brc.b2 = std::get<3>(args);
                adam_blt.b1 = std::get<2>(args);
                adam_blt.b2 = std::get<3>(args);

                adam_w.b1powt = adam_w.b1;
                adam_w.b2powt = adam_w.b2;
                adam_brc.b1powt = adam_brc.b1;
                adam_brc.b2powt = adam_brc.b2;
                adam_blt.b1powt = adam_blt.b1;
                adam_blt.b2powt = adam_blt.b2;
            }
        }

        template<typename X>
        OutputType forward(X&& input);

        template<typename X>
        InputType backward(X&& error);

        template<typename X>
        LatentType encode(X&& input);

        template<typename X>
        OutputType decode(X&& latent);

        void update(double rate);
    
    protected:
        size_t in_size, latent_size;

        Eigen::MatrixXd W;
        Eigen::VectorXd blt, brc;
        Eigen::VectorXd in, latent;
        Eigen::VectorXd zlt, zrc;
        Eigen::VectorXd alt, arc;
        Eigen::VectorXd drc, dlt;

        // Adam optimization data
        adam::AdamData<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> adam_w;
        adam::AdamData<LatentType> adam_blt;
        adam::AdamData<OutputType> adam_brc;

        template<typename X>
        LatentType errorReconstruct(X&& error);

        template<typename X>
        InputType errorLatent(X&& error);
    };
}

template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
template<typename X>
neuralnet::AutoEncoder<T, F, C>::OutputType neuralnet::AutoEncoder<T, F, C>::forward(X&& input)
{
    auto x = encode(input);
    return decode(x);
}


template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
template<typename X>
neuralnet::AutoEncoder<T, F, C>::InputType neuralnet::AutoEncoder<T, F, C>::backward(X&& error)
{
    auto x = errorReconstruct(error);
    return errorLatent(x);
}


template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
template<typename X>
neuralnet::AutoEncoder<T, F, C>::LatentType neuralnet::AutoEncoder<T, F, C>::encode(X&& input)
{
    // Save input for this pass and calculate weighted signals
    in = {input};
    zlt = blt;
    zlt += W.transpose() * input;
    // Calculate and save activation function output
    alt =  Activation<Eigen::Dynamic, T, F>::f(zlt);
    return alt;
}


template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
template<typename X>
neuralnet::AutoEncoder<T, F, C>::OutputType neuralnet::AutoEncoder<T, F, C>::decode(X&& input)
{
    // Calculate and save activation function output
    zrc = brc;
    zrc += W * input;
    arc =  Activation<Eigen::Dynamic, T, F>::f(zrc);
    return arc;
}


template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
template<typename X>
neuralnet::AutoEncoder<T, F, C>::LatentType neuralnet::AutoEncoder<T, F, C>::errorReconstruct(X&& err)
{
    // Calculate this layers error gradient
    drc = Activation<Eigen::Dynamic, T, F>::df(zrc, arc, err);
    // Calculate and return error gradient input to next layer
    return W.transpose() * drc;
}


template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
template<typename X>
neuralnet::AutoEncoder<T, F, C>::InputType neuralnet::AutoEncoder<T, F, C>::errorLatent(X&& err)
{
    // Calculate this layers error gradient
    dlt = Activation<Eigen::Dynamic, T, F>::df(zlt, alt, err);
    // Calculate and return error gradient input to next layer
    return W * dlt;
}


template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
void neuralnet::AutoEncoder<T, F, C>::update(double rate)
{
    if constexpr (C == OptimizerClass::Adam)
    {
        auto tmp = in * ((in * dlt.transpose()) + (drc * alt.transpose()));
        adam::adam_update_params(rate / 2.0, adam_w, W, tmp);
        adam::adam_update_params(rate, adam_blt, blt, dlt);
        adam::adam_update_params(rate, adam_brc, brc, drc);
    }
    else
    {
        W -= ((in * dlt.transpose()) + (drc * alt.transpose())) * (rate / 2.0);
        blt -= rate * dlt;
        brc -= rate * drc;
    }
}
#endif