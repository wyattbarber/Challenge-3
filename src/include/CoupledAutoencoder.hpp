#ifndef _COUPLEDAUTOENCODER_HPP
#define _COUPLEDAUTOENCODER_HPP

#include "DeepAutoencoder.hpp"


class CoupledAutoencoder
{
protected:
    DeepAutoencoder modelA = DeepAutoencoder(std::vector<size_t>(2, 10));
    DeepAutoencoder modelB = DeepAutoencoder(std::vector<size_t>(2, 10));
    size_t latent_size;

public:
    CoupledAutoencoder(std::vector<size_t> dims1, std::vector<size_t> dims2, size_t latent) : latent_size(latent);

    Eigen::VectorXd encode(Eigen::VectorXd input, int domain);

    Eigen::VectorXd decode(Eigen::VectorXd latent, int domain);

    std::vector<std::vector<double>> train(Eigen::MatrixXd dataA, Eigen::MatrixXd dataB, double rate, int epochs, double lambda);

    std::vector<std::vector<double>> train(Eigen::MatrixXd dataA, Eigen::MatrixXd dataB, double rate, int epochs, double lambda, double b1, double b2);
};

#endif