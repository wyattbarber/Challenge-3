#ifndef _TANH_HPP
#define _TANH_HPP
#include "Layer.hpp"
/**
 * Layer that uses Sigmoid activation function
 */
class Tanh : public Layer
{
public:
    Tanh(int in_size, int out_size) : Layer(in_size, out_size){};

    Eigen::VectorXd forward(Eigen::VectorXd input) override;

    Eigen::VectorXd error(Eigen::VectorXd err) override;
};
#endif