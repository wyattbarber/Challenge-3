#ifndef _PYSEQUENCE_HPP
#define _PYSEQUENCE_HPP

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
    template <typename T>
    class PySequence : public Model<PySequence<T>>
    {
    protected:
        std::vector<DynamicModel<T>*> layers;

    public:
        typedef Eigen::Vector<T, Eigen::Dynamic> DataType;

        /** Compose a model from a sequence of smaller models
         * 
        */
        PySequence(py::args args)
        {
            for(auto arg = args.begin(); arg != args.end(); ++arg)
            {
                void* a = arg->cast<void *>();
                layers.push_back((DynamicModel<T>*)a);
            }
        }

        /**
         * Performs one forward pass, generating output for the complete model.
         *
         * @param input data to pass to the input layer
         * @return output of the final layer
         */
        DataType forward(DataType& input);
        
        /**
         * Performs one backward pass through each layer
         *
         * @param err Output error of the model
         * @return Error gradient of the input to the model
         */
        DataType backward(DataType& error);

        void update(double rate);
    };
};

template <typename T>
neuralnet::PySequence<T>::DataType neuralnet::PySequence<T>::forward(neuralnet::PySequence<T>::DataType& input)
{
    DataType h = input;
    for (auto l : layers)
    {
        h = l->forward(h);
    }
    return h;
}


template <typename T>
neuralnet::PySequence<T>::DataType neuralnet::PySequence<T>::backward(neuralnet::PySequence<T>::DataType& err)
{
    DataType e = err;
    for (int l = layers.size() - 1; l >= 0; --l)
    {
        e = layers[l]->backward(e);
    }
    return e;
}


template <typename T>
void neuralnet::PySequence<T>::update(double rate)
{
    for(auto l = layers.begin(); l != layers.end(); ++l)
    {
        (*l)->update(rate);
    }
}



#endif
