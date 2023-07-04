#ifndef _NETWORK_HPP
#define _NETWORK_HPP
#include "Layer.hpp"
#include "ReLU.hpp"
#include "Sigmoid.hpp"
#include "Softmax.hpp"

/**
 * Network class
 *
 * This class acts as a container for Layer instances, handling forward and backward propagation through them.
 */
class Network
{
protected:
    std::vector<Layer *> layers;

public:
    Network(std::vector<int> dims, std::vector<activation::ActivationFunc> funcs);

    Eigen::VectorXd forward(Eigen::VectorXd input);
    
    std::vector<double> train(std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> outputs, double rate, int passes);

    std::vector<double> train(std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> outputs, double rate, int passes, double b1, double b2);
};

#endif