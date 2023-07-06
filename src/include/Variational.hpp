#ifndef _DEEPVARIATIONAL_HPP
#define _DEEPVARIATIONAL_HPP

#include "Autoencoder.hpp"
#include "ReLU.hpp"
#include "Sigmoid.hpp"
#include "Linear.hpp"

class VariationalAutoencoder
{
protected:
    std::vector<Autoencoder> layers;
    Sigmoid mean; 
    Sigmoid deviation;
    Sigmoid sampler;

    std::default_random_engine generator;
    std::normal_distribution<double> ndist;

public:
    VariationalAutoencoder(std::vector<size_t> dims);

    std::pair<Eigen::VectorXd, Eigen::VectorXd> encode(Eigen::VectorXd input);

    Eigen::VectorXd decode(Eigen::VectorXd latent);

    Eigen::VectorXd errorReconstruct(Eigen::VectorXd error);

    Eigen::VectorXd errorLatent(Eigen::VectorXd error_mean, Eigen::VectorXd error_deviation);

    void update(double rate);

    void update(double rate, double b1, double b2, int t);

    std::vector<double> train(Eigen::MatrixXd data, double rate, int epochs, int L);

    std::vector<double> train(Eigen::MatrixXd data, double rate, int epochs, int L, double b1, double b2);

    Eigen::VectorXd sample(Eigen::VectorXd mean, Eigen::VectorXd deviation);
};

#endif