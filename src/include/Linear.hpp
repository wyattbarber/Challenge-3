#ifndef _LINEAR_HPP
#define _LINEAR_HPP
#include "Layer.hpp"
/**
 * Layer that uses Sigmoid activation function
 */
class Linear : public Layer
{
public:
    Linear(int in_size, int out_size);

    Eigen::VectorXd forward(Eigen::VectorXd input) override;

    Eigen::VectorXd error(Eigen::VectorXd err) override;
};
#endif