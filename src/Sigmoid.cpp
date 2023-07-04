#include "include/Sigmoid.hpp"

Eigen::VectorXd Sigmoid::forward(Eigen::VectorXd input)
{
    in = input;
    z = (weights.transpose() * input) + biases;

    for (int i = 0; i < z.size(); ++i)
    {
        a(i) = 1.0 / (1.0 + std::exp(-z(i)));
    }

    return a;
}

Eigen::VectorXd Sigmoid::error(Eigen::VectorXd err)
{
    // Calculate this layers error gradient
    d = Eigen::VectorXd::Zero(d.size());
    for (int i = 0; i < d.size(); ++i)
    {
        d(i) = err(i) * a(i) * (1.0 - a(i));
    }
    return weights * d;
}