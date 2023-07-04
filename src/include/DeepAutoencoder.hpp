#ifndef _DEEPAUTOENCODER_HPP
#define _DEEPAUTOENCODER_HPP

#include "Autoencoder.hpp"

class DeepAutoencoder
{
protected:
    std::vector<Autoencoder> layers;

public:
    DeepAutoencoder(std::vector<size_t> dims);

    Eigen::VectorXd encode(Eigen::VectorXd input);

    Eigen::VectorXd decode(Eigen::VectorXd latent);

    Eigen::VectorXd errorReconstruct(Eigen::VectorXd error);

    Eigen::VectorXd errorLatent(Eigen::VectorXd error);

    void update(double rate);

    void update(double rate, double b1, double b2, int t);

    std::vector<double> train(Eigen::MatrixXd data, double rate, int epochs);

    std::vector<double> train(Eigen::MatrixXd data, double rate, int epochs, double b1, double b2);
};

#endif