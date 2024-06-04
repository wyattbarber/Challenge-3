#include "Layer.hpp"


static const double epsilon = 1e-9; /// Smallest value to allow in denominators, for stability
std::shared_ptr<Eigen::VectorXd> neuralnet::Layer<neuralnet::ActivationFunc::SoftMax>::forward(Eigen::VectorXd& input)
{
    set_z(input);
    
    for (int i = 0; i < z.size(); ++i)
    {
        a(i) = std::min(std::exp(z(i)), 1e300);
    }
    a /= abs(a.array().sum()) < epsilon ? (epsilon * (std::signbit(a.array().sum()) ? -1.0 : 1.0)) : a.array().sum();
    return std::make_shared<Eigen::VectorXd>(a);
}


std::shared_ptr<Eigen::VectorXd> neuralnet::Layer<neuralnet::ActivationFunc::SoftMax>::backward(Eigen::VectorXd& err)
{
    // Calculate this layers error gradient
    for (int i = 0; i < d.size(); ++i)
    {
        Eigen::VectorXd kd = Eigen::VectorXd::Zero(d.size());
        kd(i) = 1.0;
        d(i) = err.dot(a(i) * (kd - a));
    }
    return std::make_shared<Eigen::VectorXd>(weights * d);
}