#include "Layer.hpp"

std::shared_ptr<Eigen::VectorXd> neuralnet::Layer<neuralnet::ActivationFunc::TanH>::forward(Eigen::VectorXd& input)
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

std::shared_ptr<Eigen::VectorXd> neuralnet::Layer<neuralnet::ActivationFunc::TanH>::backward(Eigen::VectorXd& err)
{
    // Calculate this layers error gradient
    for (int i = 0; i < d.size(); ++i)
    {
        d(i) = err(i) * (1.0 - std::pow(a(i), 2.0));
    }
    return std::make_shared<Eigen::VectorXd>(weights * d);
}