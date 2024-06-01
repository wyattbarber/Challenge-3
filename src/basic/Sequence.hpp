#ifndef _SEQUENCE_HPP
#define _SEQUENCE_HPP

#include "../Model.hpp"
#include <random>
#include <vector>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace neuralnet
{
    /** Container for Layer instances, handling forward and backward propagation through them.
     */
    class Sequence : public Model
    {
    protected:
        std::vector<Model*> layers;

    public:
        Sequence(std::vector<neuralnet::Model *> layers) : layers(layers){}

        Eigen::VectorXd forward(Eigen::VectorXd input);
        
        Eigen::VectorXd backward(Eigen::VectorXd error);

        std::vector<double> train(std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> outputs, double rate, int passes);

        std::vector<double> train(std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> outputs, double rate, int passes, double b1, double b2);

        /**
         * Updates parameters of this layer based on the previously propagated error
         * @param rate learning rate
         */
        void update(double rate){};

        /**
         * Updates parameters of this layer based on the previously propagated error, using the Adam algorithm
         * @param rate learning rate
         * @param b1 first moment decay rate
         * @param b2 second moment decay rate
         * @param t current training step
         */
        void update(double rate, double b1, double b2, int t){};
    };
};

#endif
