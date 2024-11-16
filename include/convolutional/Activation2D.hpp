#ifndef _ACTIVATION2D_HPP
#define _ACTIVATION2D_HPP

#include "../Model.hpp"
#include "../optimizers/Optimizer.hpp"
#include "Activation.hpp"
#include <random>
#include <memory>

using namespace optimization;

namespace neuralnet
{
    /** Basic layer of a neural network
     *
     * @tparam F Enumerated activation function to use in this layer
     */
    template <typename T, ActivationFunc F, OptimizerClass C>
    class Layer : public Model<Layer<T, F, C>>
    {

    public:
        typedef Eigen::Vector<T, Eigen::Dynamic> InputType;
        typedef Eigen::Vector<T, Eigen::Dynamic> OutputType;

        /** Constructs a randomly initialized layer.
         *
         * Weights are initialized using He initialization, and
         * biases are initialized to 0.
         *
         * @param in_size size of the input vector to this layer
         * @param out_size size of the output vector from this layer
         */
        template <typename... Ts>
        Layer(Ts... Args)
        {
            auto args = std::tuple<Ts...>(Args...);

            int in_size = std::get<0>(args);
            int out_size = std::get<1>(args);

            // Apply he initialization
            this->weights = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Random(in_size, out_size).unaryExpr([in_size](double x)
                                                                                        { return x * std::sqrt(2.0 / static_cast<double>(in_size)); });

            this->biases = OutputType::Zero(out_size);
            this->z = OutputType::Zero(out_size);
            this->a = OutputType::Zero(out_size);
            this->d = OutputType::Zero(out_size);
            this->in = InputType::Zero(in_size);

            if constexpr (C == OptimizerClass::Adam)
            {
                adam_weights.m = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(in_size, out_size);
                adam_weights.v = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(in_size, out_size);
                adam_biases.m = OutputType::Zero(out_size);
                adam_biases.v = OutputType::Zero(out_size);

                adam_weights.b1 = std::get<2>(args);
                adam_weights.b2 = std::get<3>(args);
                adam_biases.b1 = std::get<2>(args);
                adam_biases.b2 = std::get<3>(args);

                adam_weights.b1powt = adam_weights.b1;
                adam_weights.b2powt = adam_weights.b2;
                adam_biases.b1powt = adam_biases.b1;
                adam_biases.b2powt = adam_biases.b2;
            }
        }

        template<typename X>      
        OutputType forward(X&& input);
        
        template<typename X>
        InputType backward(X&& error);

        void update(double rate);

    protected:
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> weights;
        OutputType biases;
        OutputType z;
        OutputType a;
        OutputType d;
        InputType in;

        // Data for adam optimization
        adam::AdamData<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> adam_weights;
        adam::AdamData<OutputType> adam_biases;
    };
}

template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
void neuralnet::Layer<T, F, C>::update(double rate)
{
    if constexpr (C == OptimizerClass::Adam)
    {
        auto tmp = in * d.transpose();
        adam::adam_update_params(rate, adam_weights, weights, tmp);
        adam::adam_update_params(rate, adam_biases, biases, d);
    }
    else
    {
        weights -= in * (d.transpose() * rate);
        biases -= rate * d;
    }
}

template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
template<typename X>
neuralnet::Layer<T, F, C>::OutputType neuralnet::Layer<T, F, C>::forward(X&& input)
{
    // Save input for this pass and calculate weighted signals
    in = {input};
    z = biases;
    z += (weights.transpose() * input);
    // Calculate and save activation function output
    a =  Activation<Eigen::Dynamic, T, F>::f(z);
    return a;
}

template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
template<typename X>
neuralnet::Layer<T, F, C>::InputType neuralnet::Layer<T, F, C>::backward(X&& err)
{
    // Calculate this layers error gradient
    d = Activation<Eigen::Dynamic, T, F>::df(z, a, err);
    // Calculate and return error gradient input to next layer
    return weights * d;
}

#endif
