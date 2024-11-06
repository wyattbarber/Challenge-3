#include "../include/training/Trainer.hpp"


template<class M, typename I, typename O>
std::vector<double> train(M& model, std::vector<I>& input, std::vector<O>& output, double rate, size_t epochs)
{
    training::Trainer trainer(model, input, output);
    return trainer.train(epochs, rate);
}