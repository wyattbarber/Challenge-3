#include "Layer.hpp"

std::shared_ptr<Eigen::VectorXd> neuralnet::Layer<neuralnet::ActivationFunc::Sigmoid>::forward(Eigen::VectorXd& input)
{
    set_z(input);

    for (int i = 0; i < z.size(); ++i)
    {
        a(i) = 1.0 / (1.0 + std::exp(-z(i)));
    }

    return std::make_shared<Eigen::VectorXd>(a);
}

std::shared_ptr<Eigen::VectorXd> neuralnet::Layer<neuralnet::ActivationFunc::Sigmoid>::backward(Eigen::VectorXd& err)
{
    // Calculate this layers error gradient
    d = Eigen::VectorXd::Zero(d.size());
    for (int i = 0; i < d.size(); ++i)
    {
        d(i) = err(i) * a(i) * (1.0 - a(i));
    }
    return std::make_shared<Eigen::VectorXd>(weights * d);
}