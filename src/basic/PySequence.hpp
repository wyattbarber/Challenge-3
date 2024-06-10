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
    class PySequence : public Model<Eigen::Dynamic, Eigen::Dynamic, T>
    {
    protected:
        std::vector<Model<Eigen::Dynamic, Eigen::Dynamic, T>*> layers;

    public:
        /** Compose a model from a sequence of smaller models
         * 
        */
        PySequence(py::args args)
        {
            for(auto arg = args.begin(); arg != args.end(); ++arg)
            {
                void* a = arg->cast<void *>();
                layers.push_back((Model<Eigen::Dynamic, Eigen::Dynamic, T>*)a);
            }
        }

        /**
         * Performs one forward pass, generating output for the complete model.
         *
         * @param input data to pass to the input layer
         * @return output of the final layer
         */
        Eigen::Vector<T, Eigen::Dynamic> forward(Eigen::Vector<T, Eigen::Dynamic>& input);
        
        /**
         * Performs one backward pass through each layer
         *
         * @param err Output error of the model
         * @return Error gradient of the input to the model
         */
        Eigen::Vector<T, Eigen::Dynamic> backward(Eigen::Vector<T, Eigen::Dynamic>& error);

        void update(double rate);
    };
};

template <typename T>
Eigen::Vector<T, Eigen::Dynamic> neuralnet::PySequence<T>::forward(Eigen::Vector<T, Eigen::Dynamic>& input)
{
    Eigen::VectorXd h = input;
    for (auto l : layers)
    {
        h = l->forward(h);
    }
    return h;
}


template <typename T>
Eigen::Vector<T, Eigen::Dynamic> neuralnet::PySequence<T>::backward(Eigen::Vector<T, Eigen::Dynamic>& err)
{
    Eigen::VectorXd e = err;
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
