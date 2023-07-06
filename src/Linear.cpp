#include "include/Linear.hpp"

Linear::Linear(int in_size, int out_size) : Layer(in_size, out_size){
    d = Eigen::VectorXd::Ones(d.size());
};

Eigen::VectorXd Linear::forward(Eigen::VectorXd input)
{
    in = input;
    z = (weights.transpose() * input) + biases;
    a = z;
    return a;
}

Eigen::VectorXd Linear::error(Eigen::VectorXd err)
{
    return weights * d;
}