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
    template <typename T>
    class PySequence : public Model<PySequence<T>>
    {
    protected:
        std::vector<std::shared_ptr<DynamicModel<T>>> layers;

    public:
        typedef Eigen::Vector<T, Eigen::Dynamic> InputType;
        typedef Eigen::Vector<T, Eigen::Dynamic> OutputType;

        /** Compose a model from a sequence of smaller models
         * 
        */
        PySequence(std::vector<std::shared_ptr<DynamicModel<T>>> layers) : layers{layers}
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

template <typename T>
template<typename X>
neuralnet::PySequence<T>::OutputType neuralnet::PySequence<T>::forward(X&& input)
{
    InputType h = input;
    for (auto l : layers)
    {
        h = l->forward(h);
    }
    return h;
}


template <typename T>
template<typename X>
neuralnet::PySequence<T>::InputType neuralnet::PySequence<T>::backward(X&& err)
{
    OutputType e = err;
    for (int l = layers.size() - 1; l >= 0; --l)
    {
        e = layers[l]->backward(e);
    }
    return e;
}


template <typename T>
void neuralnet::PySequence<T>::update(double rate)
{
    for(auto l : layers)
    {
        l->update(rate);
    }
}

#endif
