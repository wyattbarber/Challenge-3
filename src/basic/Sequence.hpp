#ifndef _SEQUENCE_HPP
#define _SEQUENCE_HPP

#include "../Model.hpp"
#include <random>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
namespace py = pybind11;

namespace neuralnet
{
    /** Container for Layer instances, handling forward and backward propagation through them.
     * 
     * 
     */
    class Sequence : public Model
    {
    protected:
        const std::vector<Model*> layers;

    public:
        /** Compose a model from a sequence of smaller models
         * 
         * @param Models sequence of models to compose into this one
        */
        template<class... Ts>
        Sequence(Ts&... Models): layers{(&Models)...}{}

        /**
         * Performs one forward pass, generating output for the complete model.
         *
         * @param input data to pass to the input layer
         * @return output of the final layer
         */
        Eigen::VectorXd forward(Eigen::VectorXd input);
        
        /**
         * Performs one backward pass through each layer
         *
         * @param err Output error of the model
         * @return Error gradient of the input to the model
         */
        Eigen::VectorXd backward(Eigen::VectorXd error);

        void update(double rate);
    };
};

#endif
