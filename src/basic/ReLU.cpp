#include "Layer.hpp"

template<>
Eigen::VectorXd neuralnet::Layer<neuralnet::ActivationFunc::ReLU>::forward(Eigen::VectorXd input)
{
    in = input;
    z = (weights.transpose() * input) + biases;
    for (int i = 0; i < z.size(); ++i)
    {
        if (z(i) < 0.0)
        {
            a(i) = 0.0;
        }
        else
        {
            a(i) = z(i);
        }
    }
    return a;
}

template<>
Eigen::VectorXd neuralnet::Layer<neuralnet::ActivationFunc::ReLU>::backward(Eigen::VectorXd err)
{
    // Calculate this layers error gradient
    d = Eigen::VectorXd::Zero(d.size());
    for (int i = 0; i < z.size(); ++i)
    {
        if (z(i) > 0.0)
        {
            d(i) = err(i);
        }
    }
    return weights * d;
}