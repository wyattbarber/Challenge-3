#ifndef _LAYER_HPP
#define _LAYER_HPP

#include "../Model.hpp"
#include "../optimizers/Optimizer.hpp"
#include "Activation.hpp"
#include <random>
#include <memory>
#include <pybind11/pybind11.h>
namespace py = pybind11;

using namespace optimization;

namespace neuralnet
{
    /** Basic layer of a neural network
     *
     * @tparam F Enumerated activation function to use in this layer
     */
    template <int I, int O, typename T, ActivationFunc F, OptimizerClass C>
    class Layer : public Model<I, O, T>
    {

    public:
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

            int in_size, out_size;
            if constexpr ((I == Eigen::Dynamic) || (O == Eigen::Dynamic))
            {
                in_size = std::get<0>(args);
                out_size = std::get<1>(args);
            }
            else
            {
                in_size = I;
                out_size = O;
            }

            // Apply he initialization
            this->weights = Eigen::Matrix<T, I, O>::Random(in_size, out_size).unaryExpr([in_size](double x)
                                                                                        { return x * std::sqrt(2.0 / static_cast<double>(in_size)); });

            this->biases = Eigen::Vector<T, O>::Zero(out_size);
            this->z = Eigen::Vector<T, O>::Zero(out_size);
            this->a = Eigen::Vector<T, O>::Zero(out_size);
            this->d = Eigen::Vector<T, O>::Zero(out_size);
            this->in = Eigen::Vector<T, I>::Zero(in_size);

            if constexpr (C == OptimizerClass::Adam)
            {
                adam_weights.m = Eigen::Matrix<T, I, O>::Zero(in_size, out_size);
                adam_weights.v = Eigen::Matrix<T, I, O>::Zero(in_size, out_size);
                adam_biases.m = Eigen::Vector<T, O>::Zero(out_size);
                adam_biases.v = Eigen::Vector<T, O>::Zero(out_size);

                if constexpr ((I == Eigen::Dynamic) || (O == Eigen::Dynamic))
                {
                    adam_weights.b1 = std::get<2>(args);
                    adam_weights.b2 = std::get<3>(args);
                    adam_biases.b1 = std::get<2>(args);
                    adam_biases.b2 = std::get<3>(args);
                }
                else
                {
                    adam_weights.b1 = std::get<0>(args);
                    adam_weights.b2 = std::get<1>(args);
                    adam_biases.b1 = std::get<0>(args);
                    adam_biases.b2 = std::get<1>(args);
                }
                adam_weights.b1powt = adam_weights.b1;
                adam_weights.b2powt = adam_weights.b2;
                adam_biases.b1powt = adam_biases.b1;
                adam_biases.b2powt = adam_biases.b2;
            }
        }

        Eigen::Vector<T, O> forward(Eigen::Vector<T, I> &input);

        Eigen::Vector<T, I> backward(Eigen::Vector<T, O> &error);

        void update(double rate);

    protected:
        Eigen::Matrix<T, I, O> weights;
        Eigen::Vector<T, O> biases;
        Eigen::Vector<T, O> z;
        Eigen::Vector<T, O> a;
        Eigen::Vector<T, O> d;
        Eigen::Vector<T, I> in;

        Activation<O, T, F> activation;

        // Data for adam optimization
        adam::AdamData<Eigen::Matrix<T, I, O>> adam_weights;
        adam::AdamData<Eigen::Vector<T, O>> adam_biases;
    };
}

template <int I, int O, typename T, neuralnet::ActivationFunc F, OptimizerClass C>
void neuralnet::Layer<I, O, T, F, C>::update(double rate)
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

template <int I, int O, typename T, neuralnet::ActivationFunc F, OptimizerClass C>
Eigen::Vector<T, O> neuralnet::Layer<I, O, T, F, C>::forward(Eigen::Vector<T, I> &input)
{
    // Save input for this pass and calculate weighted signals
    in = {input};
    z = biases;
    z += weights.transpose() * input;
    // Calculate and save activation function output
    a = activation.f(z);
    return a;
}

template <int I, int O, typename T, neuralnet::ActivationFunc F, OptimizerClass C>
Eigen::Vector<T, I> neuralnet::Layer<I, O, T, F, C>::backward(Eigen::Vector<T, O> &err)
{
    // Calculate this layers error gradient
    d = activation.df(z, a, err);
    // Calculate and return error gradient input to next layer
    return weights * d;
}

#endif
