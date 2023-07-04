#ifndef _RELU_HPP
#define _RELU_HPP
#include "Layer.hpp"

/**
 * Layer that uses ReLU activation function
 */
class ReLU : public Layer
{
public:
    ReLU(int in_size, int out_size) : Layer(in_size, out_size){};

    Eigen::VectorXd forward(Eigen::VectorXd input) override;

protected:
    Eigen::VectorXd error(Eigen::VectorXd err) override;
};

#endif