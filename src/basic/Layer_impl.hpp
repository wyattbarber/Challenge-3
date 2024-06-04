#ifndef _LAYER_IMPL_HPP
#define _LAYER_IMPL_HPP
#include "Layer.hpp"

template <int I, int O, typename T, neuralnet::ActivationFunc F>
void neuralnet::Layer<I, O, T, F>::update(double rate)
{
    Eigen::Matrix<T, I, O> weight_grad = in * d.transpose();
    if (is_optimized)
    {
        opt->augment_gradients(weight_grad, d);
    }
    weights -= rate * weight_grad;
    biases -= rate * d;
}

template <int I, int O, typename T, neuralnet::ActivationFunc F>
void neuralnet::Layer<I, O, T, F>::set_z(Eigen::Vector<T, I> &input)
{
    in = {input};
    z = biases;
    z += weights.transpose() * input;
}

template <int I, int O, typename T, neuralnet::ActivationFunc F>
void neuralnet::Layer<I, O, T, F>::apply_optimizer(optimization::Optimizer &opt)
{
    this->opt = opt.copy();
    if ((I != Eigen::Dynamic) && (O != Eigen::Dynamic))
        this->opt->init(I, O);
    else
        this->opt->init(weights.rows(), weights.cols());
    is_optimized = true;
}

template <int I, int O, typename T, neuralnet::ActivationFunc F>
std::shared_ptr<Eigen::Vector<T, O>> neuralnet::Layer<I, O, T, F>::forward(Eigen::Vector<T, I> &input){}

template <int I, int O, typename T, neuralnet::ActivationFunc F>
std::shared_ptr<Eigen::Vector<T, I>> neuralnet::Layer<I, O, T, F>::backward(Eigen::Vector<T, O> &err){}

/*
Linear activation function
*/
template <int I, int O, typename T>
std::shared_ptr<Eigen::Vector<T, O>> neuralnet::Layer<I, O, T, neuralnet::ActivationFunc::Linear>::forward(Eigen::Vector<T, I> &input)
{
    set_z(input);
    a = z;
    return std::make_shared<Eigen::Vector<T, O>>(a);
}

template <int I, int O, typename T>
std::shared_ptr<Eigen::Vector<T, I>> neuralnet::Layer<I, O, T, neuralnet::ActivationFunc::Linear>::backward(Eigen::Vector<T, O> &err)
{
    d = err;
    return std::make_shared<Eigen::Vector<T, I>>(weights * d);
}

/*
ReLU activation function
*/
template <int I, int O, typename T>
std::shared_ptr<Eigen::Vector<T, O>> neuralnet::Layer<I, O, T, neuralnet::ActivationFunc::ReLU>::forward(Eigen::Vector<T, I> &input)
{
    set_z(input);
    a = z;
    a.unaryExpr([](double x)
                { return (x > 0) ? x : 0.0; });
    return std::make_shared<Eigen::Vector<T, O>>(a);
}

template <int I, int O, typename T>
std::shared_ptr<Eigen::Vector<T, I>> neuralnet::Layer<I, O, T, neuralnet::ActivationFunc::ReLU>::backward(Eigen::Vector<T, O> &err)
{
    // Calculate this layers error gradient
    d = Eigen::Vector<T, O>::Zero(d.size());
    for (int i = 0; i < z.size(); ++i)
    {
        if (z(i) > 0.0)
        {
            d(i) = err(i);
        }
    }
    return std::make_shared<Eigen::Vector<T, I>>(weights * d);
}

/*
Sigmoid activation function
*/
template <int I, int O, typename T>
std::shared_ptr<Eigen::Vector<T, O>> neuralnet::Layer<I, O, T, neuralnet::ActivationFunc::Sigmoid>::forward(Eigen::Vector<T, I> &input)
{
    set_z(input);

    for (int i = 0; i < z.size(); ++i)
    {
        a(i) = 1.0 / (1.0 + std::exp(-z(i)));
    }

    return std::make_shared < Eigen::Vector<T<O>>(a);
}

template <int I, int O, typename T>
std::shared_ptr<Eigen::Vector<T, I>> neuralnet::Layer<I, O, T, neuralnet::ActivationFunc::Sigmoid>::backward(Eigen::Vector<T, O> &err)
{
    // Calculate this layers error gradient
    d = Eigen::Vector<T, O>::Zero(d.size());
    for (int i = 0; i < d.size(); ++i)
    {
        d(i) = err(i) * a(i) * (1.0 - a(i));
    }
    return std::make_shared<Eigen::Vector<T, I>>(weights * d);
}

/*
TanH activation function
*/
template <int I, int O, typename T>
std::shared_ptr<Eigen::Vector<T, O>> neuralnet::Layer<I, O, T, neuralnet::ActivationFunc::TanH>::forward(Eigen::Vector<T, I> &input)
{
    set_z(input);
    for (int i = 0; i < z.size(); ++i)
    {
        double ex = std::exp(z(i));
        double nex = std::exp(-z(i));
        a(i) = (ex - nex) / (ex + nex);
    }
    return std::make_shared<Eigen::VectorXd>(a);
}

template <int I, int O, typename T>
std::shared_ptr<Eigen::Vector<T, I>> neuralnet::Layer<I, O, T, neuralnet::ActivationFunc::TanH>::backward(Eigen::Vector<T, O> &err)
{
    // Calculate this layers error gradient
    for (int i = 0; i < d.size(); ++i)
    {
        d(i) = err(i) * (1.0 - std::pow(a(i), 2.0));
    }
    return std::make_shared<Eigen::Vector<T, I>>(weights * d);
}

/*
Softmax activation function
*/
static const double epsilon = 1e-9; /// Smallest value to allow in denominators, for stability
template <int I, int O, typename T>
std::shared_ptr<Eigen::Vector<T, O>> neuralnet::Layer<I, O, T, neuralnet::ActivationFunc::SoftMax>::forward(Eigen::Vector<T, I> &input)
{
    set_z(input);

    for (int i = 0; i < z.size(); ++i)
    {
        a(i) = std::min(std::exp(z(i)), 1e300);
    }
    a /= abs(a.array().sum()) < epsilon ? (epsilon * (std::signbit(a.array().sum()) ? -1.0 : 1.0)) : a.array().sum();
    return std::make_shared<Eigen::Vector<T, O>>(a);
}

template <int I, int O, typename T>
std::shared_ptr<Eigen::Vector<T, I>> neuralnet::Layer<I, O, T, neuralnet::ActivationFunc::SoftMax>::backward(Eigen::Vector<T, O> &err)
{
    // Calculate this layers error gradient
    for (int i = 0; i < d.size(); ++i)
    {
        Eigen::Vector<T, O> kd = Eigen::Vector<T, O>::Zero(d.size());
        kd(i) = 1.0;
        d(i) = err.dot(a(i) * (kd - a));
    }
    return std::make_shared<Eigen::Vector<T, I>>(weights * d);
}
#endif