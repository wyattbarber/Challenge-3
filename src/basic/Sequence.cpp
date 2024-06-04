#include "Sequence.hpp"


std::shared_ptr<Eigen::VectorXd> neuralnet::Sequence::forward(Eigen::VectorXd& input)
{
    std::shared_ptr<Eigen::VectorXd> h = std::make_shared<Eigen::VectorXd>(input);
    for (auto l : layers)
    {
        h = l->forward(*h);
    }
    return h;
}


std::shared_ptr<Eigen::VectorXd> neuralnet::Sequence::backward(Eigen::VectorXd& err)
{
    std::shared_ptr<Eigen::VectorXd> e = std::make_shared<Eigen::VectorXd>(err);
    for (int l = layers.size() - 1; l >= 0; --l)
    {
        e = layers[l]->backward(*e);
    }
    return e;
}



void neuralnet::Sequence::update(double rate)
{
    for(auto l = layers.begin(); l != layers.end(); ++l)
    {
        (*l)->update(rate);
    }
}


void neuralnet::Sequence::apply_optimizer(optimization::Optimizer& opt)
{
    for(auto l = layers.begin(); l != layers.end(); ++l)
    {
        (*l)->apply_optimizer(opt);
    } 
}