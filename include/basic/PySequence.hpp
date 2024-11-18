#ifndef _PYSEQUENCE_HPP
#define _PYSEQUENCE_HPP

#include "../Model.hpp"
#include <random>
#include <vector>

namespace neuralnet
{
    /** Container for Layer instances, handling forward and backward propagation through them.
     * 
     * 
     */
    template <typename H>
    class PySequence : public Model<PySequence<H>>
    {
    protected:
        std::vector<std::shared_ptr<H>> layers;

    public:
        typedef H::InputType InputType;
        typedef H::OutputType OutputType;

        /** Compose a model from a sequence of smaller models
         * 
        */
        PySequence(std::vector<std::shared_ptr<H>> layers) : layers{layers}
        {
        }

        /**
         * Performs one forward pass, generating output for the complete model.
         *
         * @param input data to pass to the input layer
         * @return output of the final layer
         */
        template<typename X>
        OutputType forward(X&& input);
        
        /**
         * Performs one backward pass through each layer
         *
         * @param err Output error of the model
         * @return Error gradient of the input to the model
         */
        template<typename X>
        InputType backward(X&& error);

        void update(double rate);
    };
};

template <typename H>
template<typename X>
neuralnet::PySequence<H>::OutputType neuralnet::PySequence<H>::forward(X&& input)
{
    InputType h = input;
    for (auto l : layers)
    {
        h = l->forward(h);
    }
    return h;
}


template <typename H>
template<typename X>
neuralnet::PySequence<H>::InputType neuralnet::PySequence<H>::backward(X&& err)
{
    OutputType e = err;
    for (int l = layers.size() - 1; l >= 0; --l)
    {
        e = layers[l]->backward(e);
    }
    return e;
}


template <typename H>
void neuralnet::PySequence<H>::update(double rate)
{
    for(auto l : layers)
    {
        l->update(rate);
    }
}

#endif
