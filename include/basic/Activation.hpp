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
        template <typename X> static Eigen::Vector<T, N> f(X&& input);
        template <typename X, typename Y, typename Z> static Eigen::Vector<T, N> df(X&& z, Y&& activation, Z&& error);
    };

    // Specialization definitions
    template <int N, typename T>
    class Activation<N, T, ActivationFunc::Linear>
    {
    public:
        template <typename X> static Eigen::Vector<T, N> f(X&& input);
        template <typename X, typename Y, typename Z> static Eigen::Vector<T, N> df(X&& z, Y&& activation, Z&& error);
    };

    template <int N, typename T>
    class Activation<N, T, ActivationFunc::ReLU>
    {
    public:
        template <typename X> static Eigen::Vector<T, N> f(X&& input);
        template <typename X, typename Y, typename Z> static Eigen::Vector<T, N> df(X&& z, Y&& activation, Z&& error);
    };

    template <int N, typename T>
    class Activation<N, T, ActivationFunc::Sigmoid>
    {
    public:
        template <typename X> static Eigen::Vector<T, N> f(X&& input);
        template <typename X, typename Y, typename Z> static Eigen::Vector<T, N> df(X&& z, Y&& activation, Z&& error);
    };

    template <int N, typename T>
    class Activation<N, T, ActivationFunc::TanH>
    {
    public:
        template <typename X> static Eigen::Vector<T, N> f(X&& input);
        template <typename X, typename Y, typename Z> static Eigen::Vector<T, N> df(X&& z, Y&& activation, Z&& error);
    };

    template <int N, typename T>
    class Activation<N, T, ActivationFunc::SoftMax>
    {
    public:
        template <typename X> static Eigen::Vector<T, N> f(X&& input);
        template <typename X, typename Y, typename Z> static Eigen::Vector<T, N> df(X&& z, Y&& activation, Z&& error);
    };
};

/*
Linear activation function
*/
template <int N, typename T>
template <typename X> 
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::Linear>::f(X&& input)
{
    return input;
}

template <int N, typename T>
template <typename X, typename Y, typename Z> 
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::Linear>::df(X&& z, Y&& activation, Z&& error)
{
    return error;
}

/*
ReLU activation function
*/
template <int N, typename T>
template <typename X> 
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::ReLU>::f(X&& input)
{
    Eigen::Vector<T, N> out = Eigen::Vector<T, N>(input.size());
    for (size_t i = 0; i < input.size(); ++i)
        out(i) = std::max(T(0), input(i));
    return out;
}

template <int N, typename T>
template <typename X, typename Y, typename Z> 
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::ReLU>::df(X&& z, Y&& activation, Z&& error)
{
    Eigen::Vector<T, N> out = Eigen::Vector<T, N>(activation.size());
    for (int i = 0; i < activation.size(); ++i)
    {
        out(i) = activation(i) > 0 ? error(i) : T(0);
    }
    return out;
}

/*
Sigmoid activation function
*/
template <int N, typename T>
template <typename X> 
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::Sigmoid>::f(X&& input)
{
    Eigen::Vector<T, N> out = Eigen::Vector<T, N>(input.size());
    for (int i = 0; i < input.size(); ++i)
    {
        out(i) = T(1) / (T(1) + std::exp(-input(i)));
    }

    return out;
}

template <int N, typename T>
template <typename X, typename Y, typename Z>  
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::Sigmoid>::df(X&& z, Y&& activation, Z&& error)
{
    // Calculate this layers error gradient
    Eigen::Vector<T, N> out = Eigen::Vector<T, N>(activation.size());
    for (int i = 0; i < activation.size(); ++i)
    {
        out(i) = activation(i) * (T(1) - activation(i));
    }
    return out.cwiseProduct(error);
}

/*
TanH activation function
*/
template <int N, typename T>
template <typename X> 
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::TanH>::f(X&& input)
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
template <typename X, typename Y, typename Z> 
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::TanH>::df(X&& z, Y&& activation, Z&& error)
{
    Eigen::Vector<T, N> out = Eigen::Vector<T, N>(activation);
    for (int i = 0; i < activation.size(); ++i)
    {
        out(i) = T(1) - (activation(i) * activation(i));
    }
    return out.cwiseProduct(error);
}

/*
Softmax activation function
*/
static const double epsilon = 1e-9; /// Smallest value to allow in denominators, for stability
template <int N, typename T>
template <typename X> 
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::SoftMax>::f(X&& input)
{
    Eigen::Vector<T, N> out = Eigen::Vector<T, N>(input.size());
    for (int i = 0; i < input.size(); ++i)
    {
        out(i) = std::min(std::exp(input(i)), Eigen::NumTraits<T>::highest()); // Prevent exploding values
    }
    double sum = out.array().sum();
    out /= abs(sum) < epsilon ? (epsilon * (std::signbit(sum) ? -T(1) : T(1))) : sum;
    return out;
}

template <int N, typename T>
template <typename X, typename Y, typename Z> 
Eigen::Vector<T, N> neuralnet::Activation<N, T, neuralnet::ActivationFunc::SoftMax>::df(X&& z, Y&& activation, Z&& error)
{
    Eigen::Vector<T, N> out = Eigen::Vector<T, N>(activation.size());
    Eigen::Vector<T, N> kd = Eigen::Vector<T, N>::Zero(activation.size());
    for (int i = 0; i < activation.size(); ++i)
    {
        kd(i) = T(1); // Kronecker delta, dz_i/dz_j is 1 for i==j, 0 for all others
        out(i) = error.dot(activation(i) * (kd - activation));
        kd(i) = T(0); // Reset dz_i/dz_j
    }
    return out;
}

#endif