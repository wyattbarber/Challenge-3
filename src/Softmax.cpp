
#include "include/Softmax.hpp"

Eigen::VectorXd SoftMax::forward(Eigen::VectorXd input)
{
    in = input;
    z = (weights.transpose() * input) + biases;

    for (int i = 0; i < z.size(); ++i)
    {
        a(i) = std::min(std::exp(z(i)), 1e300);
    }

    a /= (a.array().sum() + 0.0000001);

    return a;
}

Eigen::VectorXd SoftMax::error(Eigen::VectorXd err)
{
    // Calculate this layers error gradient
    for (int i = 0; i < d.size(); ++i)
    {
        Eigen::VectorXd kd = Eigen::VectorXd::Zero(d.size());
        kd(i) = 1.0;
        d(i) = err.dot(a(i) * (kd - a));
    }
    return weights * d;
}