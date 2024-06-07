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

    /** Applies templated activation functions
     *
     */
    template <int N, typename T, ActivationFunc F>
    class Activation
    {
    public:
        Eigen::Vector<T, N> f(Eigen::Vector<T, N> &input);
        Eigen::Vector<T, N> df(Eigen::Vector<T, N> &z, Eigen::Vector<T, N> &activation, Eigen::Vector<T, N> &error);
    };

    // Specialization definitions
    template <int N, typename T>
    class Activation<N, T, ActivationFunc::Linear>
    {
    public:
        Eigen::Vector<T, N> f(Eigen::Vector<T, N> &input);
        Eigen::Vector<T, N> df(Eigen::Vector<T, N> &z, Eigen::Vector<T, N> &activation, Eigen::Vector<T, N> &error);
    };

    template <int N, typename T>
    class Activation<N, T, ActivationFunc::ReLU>
    {
    public:
        Eigen::Vector<T, N> f(Eigen::Vector<T, N> &input);
        Eigen::Vector<T, N> df(Eigen::Vector<T, N> &z, Eigen::Vector<T, N> &activation, Eigen::Vector<T, N> &error);
    };

    template <int N, typename T>
    class Activation<N, T, ActivationFunc::Sigmoid>
    {
    public:
        Eigen::Vector<T, N> f(Eigen::Vector<T, N> &input);
        Eigen::Vector<T, N> df(Eigen::Vector<T, N> &z, Eigen::Vector<T, N> &activation, Eigen::Vector<T, N> &error);
    };

    template <int N, typename T>
    class Activation<N, T, ActivationFunc::TanH>
    {
    public:
        Eigen::Vector<T, N> f(Eigen::Vector<T, N> &input);
        Eigen::Vector<T, N> df(Eigen::Vector<T, N> &z, Eigen::Vector<T, N> &activation, Eigen::Vector<T, N> &error);
    };

    template <int N, typename T>
    class Activation<N, T, ActivationFunc::SoftMax>
    {
    public:
        Eigen::Vector<T, N> f(Eigen::Vector<T, N> &input);
        Eigen::Vector<T, N> df(Eigen::Vector<T, N> &z, Eigen::Vector<T, N> &activation, Eigen::Vector<T, N> &error);
    };
};

/*
Linear activation function
*/
template <int N, typename T>
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::Linear>::f(Eigen::Vector<T, N> &input)
{
    return input;
}

template <int N, typename T>
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::Linear>::df(Eigen::Vector<T, N> &z, Eigen::Vector<T, N> &activation, Eigen::Vector<T, N> &error)
{
    return error;
}

/*
ReLU activation function
*/
template <int N, typename T>
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::ReLU>::f(Eigen::Vector<T, N> &input)
{
    Eigen::Vector<T, N> out = Eigen::Vector<T, N>(input.size());
    for (size_t i = 0; i < input.size(); ++i)
        out(i) = std::max(0.0, input(i));
    return out;
}

template <int N, typename T>
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::ReLU>::df(Eigen::Vector<T, N> &z, Eigen::Vector<T, N> &activation, Eigen::Vector<T, N> &error)
{
    Eigen::Vector<T, N> out = Eigen::Vector<T, N>(activation.size());
    for (int i = 0; i < activation.size(); ++i)
    {
        out(i) = activation(i) > 0 ? error(i) : 0.0;
    }
    return out;
}

/*
Sigmoid activation function
*/
template <int N, typename T>
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::Sigmoid>::f(Eigen::Vector<T, N> &input)
{
    Eigen::Vector<T, N> out = Eigen::Vector<T, N>(input.size());
    for (int i = 0; i < input.size(); ++i)
    {
        out(i) = 1.0 / (1.0 + std::exp(-input(i)));
    }

    return out;
}

template <int N, typename T>
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::Sigmoid>::df(Eigen::Vector<T, N> &z, Eigen::Vector<T, N> &activation, Eigen::Vector<T, N> &error)
{
    // Calculate this layers error gradient
    Eigen::Vector<T, N> out = Eigen::Vector<T, N>(activation.size());
    for (int i = 0; i < activation.size(); ++i)
    {
        out(i) = activation(i) * (1.0 - activation(i));
    }
    return out.cwiseProduct(error);
}

/*
TanH activation function
*/
template <int N, typename T>
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::TanH>::f(Eigen::Vector<T, N> &input)
{
    Eigen::Vector<T, N> out = Eigen::Vector<T, N>(input.size());
    for (int i = 0; i < input.size(); ++i)
    {
        double ex = std::exp(input(i));
        double nex = std::exp(-input(i));
        out(i) = (ex - nex) / (ex + nex);
    }
    return out;
}

template <int N, typename T>
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::TanH>::df(Eigen::Vector<T, N> &z, Eigen::Vector<T, N> &activation, Eigen::Vector<T, N> &error)
{
    Eigen::Vector<T, N> out = Eigen::Vector<T, N>(activation);
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
template <int N, typename T>
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::SoftMax>::f(Eigen::Vector<T, N> &input)
{
    Eigen::Vector<T, N> out = Eigen::Vector<T, N>(input.size());
    for (int i = 0; i < input.size(); ++i)
    {
        out(i) = std::min(std::exp(input(i)), 1e300); // Prevent exploding values
    }
    double sum = out.array().sum();
    out /= abs(sum) < epsilon ? (epsilon * (std::signbit(sum) ? -1.0 : 1.0)) : sum;
    return out;
}

template <int N, typename T>
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::SoftMax>::df(Eigen::Vector<T, N> &z, Eigen::Vector<T, N> &activation, Eigen::Vector<T, N> &error)
{
    Eigen::Vector<T, N> out = Eigen::Vector<T, N>(activation.size());
    Eigen::Vector<T, N> kd = Eigen::Vector<T, N>::Zero(activation.size());
    for (int i = 0; i < activation.size(); ++i)
    {
        kd(i) = 1.0; // Kronecker delta, dz_i/dz_j is 1 for i==j, 0 for all others
        out(i) = error.dot(activation(i) * (kd - activation));
        kd(i) = 0.0; // Reset dz_i/dz_j
    }
    return out;
}
#endif