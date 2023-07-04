#ifndef _SOFTMAX_HPP
#define _SOFTMAX_HPP
#include "Layer.hpp"

/**
 * Layer that uses SoftMax activation function
 */
class SoftMax : public Layer
{
public:
    SoftMax(int in_size, int out_size) : Layer(in_size, out_size){};

    Eigen::VectorXd forward(Eigen::VectorXd input);

    Eigen::VectorXd error(Eigen::VectorXd err);
};


#endif