#include "Layer.hpp"


template<>
std::shared_ptr<Eigen::VectorXd> neuralnet::Layer<neuralnet::ActivationFunc::Linear>::forward(Eigen::VectorXd& input)
{
    set_z(input);
    a = z;
    return  std::make_shared<Eigen::VectorXd>(a);
}

template<>
std::shared_ptr<Eigen::VectorXd> neuralnet::Layer<neuralnet::ActivationFunc::Linear>::backward(Eigen::VectorXd& err)
{
    d = err;
    return std::make_shared<Eigen::VectorXd>(weights * d);
}