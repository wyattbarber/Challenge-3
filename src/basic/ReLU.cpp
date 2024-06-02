#include "Layer.hpp"

template<>
Eigen::VectorXd neuralnet::Layer<neuralnet::ActivationFunc::ReLU>::forward(Eigen::VectorXd input)
{
    set_z(input);
    a = z;
    a.unaryExpr([](double x){ return (x > 0) ? x : 0.0; });
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