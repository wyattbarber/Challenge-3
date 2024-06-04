#include "Sequence.hpp"

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