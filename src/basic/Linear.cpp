#include "Layer.hpp"


template<>
Eigen::VectorXd neuralnet::Layer<neuralnet::ActivationFunc::Linear>::forward(Eigen::VectorXd input)
{
    set_z(input);
    a = z;
    return a;
}

template<>
Eigen::VectorXd neuralnet::Layer<neuralnet::ActivationFunc::Linear>::backward(Eigen::VectorXd err)
{
    return weights * d;
}