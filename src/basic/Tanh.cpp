#include "Layer.hpp"

Eigen::VectorXd neuralnet::Layer<neuralnet::ActivationFunc::TanH>::forward(Eigen::VectorXd input)
{
    in = input;
    z = (weights.transpose() * input) + biases;
    a = z;
    a.unaryExpr([](double x){
        double ex = std::exp(x);
        double nex = std::exp(-x);
        return (ex - nex) / (ex + nex);
    });
    return a;
}

Eigen::VectorXd neuralnet::Layer<neuralnet::ActivationFunc::TanH>::backward(Eigen::VectorXd err)
{
    // Calculate this layers error gradient
    d = Eigen::VectorXd::Zero(d.size());
    for (int i = 0; i < d.size(); ++i)
    {
        d(i) = err(i) * (1.0 - std::pow(a(i), 2.0));
    }
    return weights * d;
}