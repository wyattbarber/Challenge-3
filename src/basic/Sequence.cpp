#include "Sequence.hpp"


Eigen::VectorXd neuralnet::Sequence::forward(Eigen::VectorXd input)
{
    std::vector<Eigen::VectorXd> a;
    a.push_back(input);
    for (int l = 0; l < layers.size(); ++l)
    {
        Eigen::VectorXd h = layers[l]->forward(a.back());
        a.push_back(h);
    }
    return a.back();
}


Eigen::VectorXd neuralnet::Sequence::backward(Eigen::VectorXd err)
{
    std::vector<Eigen::VectorXd> errors;
    errors.push_back(err);

    for (int l = layers.size() - 1; l >= 0; --l)
    {
        errors.push_back(layers[l]->backward(errors.back()));
    }

    return errors.back();
}



void neuralnet::Sequence::update(double rate)
{
    for(auto l = layers.begin(); l != layers.end(); ++l)
    {
        (*l)->update(rate);
    }
}