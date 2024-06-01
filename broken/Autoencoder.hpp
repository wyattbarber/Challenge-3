#ifndef _AUTOENCODER_HPP
#define _AUTOENCODER_HPP

#include <Eigen/Dense>
#include <pybind11/pybind11.h>
namespace py = pybind11;

class Autoencoder
{
protected:
    size_t in_size, latent_size;

    Eigen::MatrixXd W;
    Eigen::VectorXd blt, brc;
    Eigen::VectorXd in;
    Eigen::VectorXd alt, arc;
    Eigen::VectorXd drc, dlt;

    // Adam optimization data
    Eigen::MatrixXd m, v;
    Eigen::VectorXd mblt, vblt, mbrc, vbrc;

public:
    Autoencoder(size_t input, size_t latent);

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