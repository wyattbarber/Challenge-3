#include "Layer.hpp"

template<>
std::shared_ptr<Eigen::VectorXd> neuralnet::Layer<neuralnet::ActivationFunc::ReLU>::forward(Eigen::VectorXd& input)
{
    set_z(input);
    a = z;
    a.unaryExpr([](double x){ return (x > 0) ? x : 0.0; });
    return std::make_shared<Eigen::VectorXd>(a);
}

template<>
std::shared_ptr<Eigen::VectorXd> neuralnet::Layer<neuralnet::ActivationFunc::ReLU>::backward(Eigen::VectorXd& err)
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
    return std::make_shared<Eigen::VectorXd>(weights * d);
}