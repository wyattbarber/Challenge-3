#include "include/DenoisingAutoencoder.hpp"

DenoisingAutoencoder::DenoisingAutoencoder(std::vector<size_t> dims) : DeepAutoencoder(dims){}

std::vector<double> DenoisingAutoencoder::train(Eigen::MatrixXd data_in, Eigen::MatrixXd data_out, double rate, int epochs)
{
    std::vector<double> out(epochs);

    // Calcualte total data magnitude for normalizing errors
    double data_norm = 0.0;
    for (int i = 0; i < data_out.cols(); ++i)
    {
        data_norm += data_out.col(i).norm();
    }

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double e = 0.0;
        for (int i = 0; i < data_in.cols(); ++i)
        {
            // Encode and decode a datapoint, calculate reconstuction error
            Eigen::VectorXd error = decode(encode(data_in.col(i))) - data_out.col(i);
            e += error.norm();

            // Backpropagate errors
            errorLatent(errorReconstruct(error));

            // Update parameters
            update(rate);
        }
        out[epoch] = e / data_norm;
    }
    return out;
}

std::vector<double> DenoisingAutoencoder::train(Eigen::MatrixXd data_in, Eigen::MatrixXd data_out, double rate, int epochs, double b1, double b2)
{
    std::vector<double> out(epochs);

    // Calcualte total data magnitude for normalizing errors
    double data_norm = 0.0;
    for (int i = 0; i < data_out.cols(); ++i)
    {
        data_norm += data_out.col(i).norm();
    }

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double e = 0.0;
        for (int i = 0; i < data_in.cols(); ++i)
        {
            // Encode and decode a datapoint, calculate reconstuction error
            Eigen::VectorXd error = decode(encode(data_in.col(i))) - data_out.col(i);
            e += error.norm();

            // Backpropagate errors
            errorLatent(errorReconstruct(error));

            // Update parameters
            update(rate, b1, b2, (epoch * data_in.cols()) + i + 1);
        }
        out[epoch] = e / data_norm;
    }
    return out;
}