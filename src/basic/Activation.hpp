#ifndef _ACTIVATION_HPP
#define _ACTIVATION_HPP

#include <Eigen/Dense>
#include <cmath>

namespace neuralnet
{

    /** Supported activation functions for simple layers
     *
     */
    enum class ActivationFunc
    {
        ReLU,
        Sigmoid,
        SoftMax,
        TanH,
        Linear
    };

    template <ActivationFunc F>
    Eigen::VectorXd activation(Eigen::VectorXd &input);

    template <ActivationFunc F>
    Eigen::VectorXd d_activation(Eigen::VectorXd &z, Eigen::VectorXd &activation, Eigen::VectorXd &error);
}

/*
Linear activation function
*/
template <>
Eigen::VectorXd neuralnet::activation<neuralnet::ActivationFunc::Linear>(Eigen::VectorXd &input)
{
    return input;
}

template <>
Eigen::VectorXd neuralnet::d_activation<neuralnet::ActivationFunc::Linear>(Eigen::VectorXd &z, Eigen::VectorXd &activation, Eigen::VectorXd &error)
{
    return error;
}

/*
ReLU activation function
*/
template <>
Eigen::VectorXd neuralnet::activation<neuralnet::ActivationFunc::ReLU>(Eigen::VectorXd &input)
{
    Eigen::VectorXd out = Eigen::VectorXd(input.size());
    for (size_t i = 0; i < input.size(); ++i)
        out(i) = std::max(0.0, input(i));
    return out;
}

template <>
Eigen::VectorXd neuralnet::d_activation<neuralnet::ActivationFunc::ReLU>(Eigen::VectorXd &z, Eigen::VectorXd &activation, Eigen::VectorXd &error)
{
    Eigen::VectorXd out = Eigen::VectorXd(activation.size());
    for (int i = 0; i < activation.size(); ++i)
    {
        out(i) = activation(i) > 0 ? error(i) : 0.0;
    }
    return out;
}

/*
Sigmoid activation function
*/
template <>
Eigen::VectorXd neuralnet::activation<neuralnet::ActivationFunc::Sigmoid>(Eigen::VectorXd &input)
{
    Eigen::VectorXd out = Eigen::VectorXd(input.size());
    for (int i = 0; i < input.size(); ++i)
    {
        out(i) = 1.0 / (1.0 + std::exp(-input(i)));
    }

    return out;
}

template <>
Eigen::VectorXd neuralnet::d_activation<neuralnet::ActivationFunc::Sigmoid>(Eigen::VectorXd &z, Eigen::VectorXd &activation, Eigen::VectorXd &error)
{
    // Calculate this layers error gradient
    Eigen::VectorXd out = Eigen::VectorXd(activation.size());
    for (int i = 0; i < activation.size(); ++i)
    {
        out(i) = activation(i) * (1.0 - activation(i));
    }
    return out.cwiseProduct(error);
}

/*
TanH activation function
*/
template <>
Eigen::VectorXd neuralnet::activation<neuralnet::ActivationFunc::TanH>(Eigen::VectorXd &input)
{
    Eigen::VectorXd out = Eigen::VectorXd(input.size());
    for (int i = 0; i < input.size(); ++i)
    {
        double ex = std::exp(input(i));
        double nex = std::exp(-input(i));
        out(i) = (ex - nex) / (ex + nex);
    }
    return out;
}

template <>
Eigen::VectorXd neuralnet::d_activation<neuralnet::ActivationFunc::TanH>(Eigen::VectorXd &z, Eigen::VectorXd &activation, Eigen::VectorXd &error)
{
    Eigen::VectorXd out = Eigen::VectorXd(activation);
    for (int i = 0; i < activation.size(); ++i)
    {
        out(i) = 1.0 - (activation(i) * activation(i));
    }
    return out.cwiseProduct(error);
}

/*
Softmax activation function
*/
static const double epsilon = 1e-9; /// Smallest value to allow in denominators, for stability
template <>
Eigen::VectorXd neuralnet::activation<neuralnet::ActivationFunc::SoftMax>(Eigen::VectorXd &input)
{
    Eigen::VectorXd out = Eigen::VectorXd(input.size());
    for (int i = 0; i < input.size(); ++i)
    {
        out(i) = std::min(std::exp(input(i)), 1e300); // Prevent exploding values
    }
    double sum = out.array().sum();
    out /= abs(sum) < epsilon ? (epsilon * (std::signbit(sum) ? -1.0 : 1.0)) : sum;
    return out;
}

template <>
Eigen::VectorXd neuralnet::d_activation<neuralnet::ActivationFunc::SoftMax>(Eigen::VectorXd &z, Eigen::VectorXd &activation, Eigen::VectorXd &error)
{
    Eigen::VectorXd out = Eigen::VectorXd(activation.size());
    Eigen::VectorXd kd = Eigen::VectorXd::Zero(activation.size());
    for (int i = 0; i < activation.size(); ++i)
    {
        kd(i) = 1.0; // Kronecker delta, dz_i/dz_j is 1 for i==j, 0 for all others
        out(i) = error.dot(activation(i) * (kd - activation));
        kd(i) = 0.0; // Reset dz_i/dz_j
    }
    return out;
}
#endif