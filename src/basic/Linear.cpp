#include "Layer.hpp"


template<>
Eigen::VectorXd neuralnet::Layer<neuralnet::ActivationFunc::Linear>::forward(Eigen::VectorXd input)
{
    in = input;
    z = (weights.transpose() * input) + biases;
    a = z;
    return a;
}

template<>
Eigen::VectorXd neuralnet::Layer<neuralnet::ActivationFunc::Linear>::backward(Eigen::VectorXd err)
{
    return weights * d;
}