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
    template <int I, int O, typename T>
    class Sequence : public Model<I, O, T>
    {
    protected:
        std::vector<Model*> layers;

    public:
        /** Compose a model from a sequence of smaller models
         * 
        */
        Sequence(py::args args)
        {
            for(auto arg = args.begin(); arg != args.end(); ++arg)
            {
                void* a = arg->cast<void *>();
                layers.push_back((Model*)a);
            }
        }
        Sequence(std::vector<Model*> args) : layers(args){}

        /**
         * Performs one forward pass, generating output for the complete model.
         *
         * @param input data to pass to the input layer
         * @return output of the final layer
         */
        std::shared_ptr<Eigen::Vector<T, O>> forward(Eigen::Vector<T, I>& input);
        
        /**
         * Performs one backward pass through each layer
         *
         * @param err Output error of the model
         * @return Error gradient of the input to the model
         */
        std::shared_ptr<Eigen::Vector<T, I>> backward(Eigen::Vector<T, O>& error);

        void update(double rate);

        void apply_optimizer(optimization::Optimizer& opt);
    };
};

template <int I, int O, typename T>
std::shared_ptr<Eigen::Vector<T, O>> neuralnet::Sequence<I, O, T>::forward(Eigen::Vector<T, I>& input)
{
    std::shared_ptr<Eigen::VectorXd> h = std::make_shared<Eigen::VectorXd>(input);
    for (auto l : layers)
    {
        h = l->forward(*h);
    }
    return h;
}


template <int I, int O, typename T>
std::shared_ptr<Eigen::Vector<T, I>> neuralnet::Sequence<I, O, T>::backward(Eigen::Vector<T, O>& err)
{
    std::shared_ptr<Eigen::VectorXd> e = std::make_shared<Eigen::VectorXd>(err);
    for (int l = layers.size() - 1; l >= 0; --l)
    {
        e = layers[l]->backward(*e);
    }
    return e;
}


template <int I, int O, typename T>
void neuralnet::Sequence<I, O, T>::update(double rate)
{
    for(auto l = layers.begin(); l != layers.end(); ++l)
    {
        (*l)->update(rate);
    }
}


template <int I, int O, typename T>
void neuralnet::Sequence<I, O, T>::apply_optimizer(optimization::Optimizer& opt)
{
    for(auto l = layers.begin(); l != layers.end(); ++l)
    {
        (*l)->apply_optimizer(opt);
    } 
}


#endif
