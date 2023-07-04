#ifndef _SIGMOID_HPP
#define _SIGMOID_HPP
#include "Layer.hpp"
/**
 * Layer that uses Sigmoid activation function
 */
class Sigmoid : public Layer
{
public:
    Sigmoid(int in_size, int out_size) : Layer(in_size, out_size){};

    Eigen::VectorXd forward(Eigen::VectorXd input) override;

    Eigen::VectorXd error(Eigen::VectorXd err) override;
};
#endif