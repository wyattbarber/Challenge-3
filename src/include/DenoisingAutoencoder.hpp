#ifndef _DENOISINGAUTOENCODER_HPP
#define _DENOISINGAUTOENCODER_HPP

#include "DeepAutoencoder.hpp"

class DenoisingAutoencoder : public DeepAutoencoder
{
protected:
    std::vector<Autoencoder> layers;

public:
    DenoisingAutoencoder(std::vector<size_t> dims);

    std::vector<double> train(Eigen::MatrixXd data_in, Eigen::MatrixXd data_out, double rate, int epochs);

    std::vector<double> train(Eigen::MatrixXd data_in, Eigen::MatrixXd data_out, double rate, int epochs, double b1, double b2);
};

#endif