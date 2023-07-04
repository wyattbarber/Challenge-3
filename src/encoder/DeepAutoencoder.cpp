#include "../include/DeepAutoencoder.hpp"

DeepAutoencoder::DeepAutoencoder(std::vector<size_t> dims)
{
    for (int i = 1; i < dims.size(); ++i)
    {
        layers.push_back(Autoencoder(dims[i - 1], dims[i]));
    }
}

Eigen::VectorXd DeepAutoencoder::encode(Eigen::VectorXd input)
{
    std::vector<Eigen::VectorXd> hidden = {input};
    for (int j = 0; j < layers.size(); ++j)
        hidden.push_back(layers[j].encode(hidden.back()));
    return hidden.back();
}

Eigen::VectorXd DeepAutoencoder::decode(Eigen::VectorXd latent)
{
    std::vector<Eigen::VectorXd> hidden = {latent};
    for (int j = layers.size() - 1; j >= 0; --j)
        hidden.push_back(layers[j].decode(hidden.back()));
    return hidden.back();
}

Eigen::VectorXd DeepAutoencoder::errorReconstruct(Eigen::VectorXd error)
{
    std::vector<Eigen::VectorXd> errors = {error};
    for (int j = 0; j < layers.size(); ++j)
        errors.push_back(layers[j].errorReconstruct(errors.back()));
    return errors.back();
}

Eigen::VectorXd DeepAutoencoder::errorLatent(Eigen::VectorXd error)
{
    std::vector<Eigen::VectorXd> errors = {error};
    for (int j = layers.size() - 1; j >= 0; --j)
        errors.push_back(layers[j].errorLatent(errors.back()));
    return errors.back();
}

void DeepAutoencoder::update(double rate)
{
    for (auto l = layers.begin(); l != layers.end(); ++l)
    {
        l->update(rate);
    }
}

void DeepAutoencoder::update(double rate, double b1, double b2, int t)
{
    for (auto l = layers.begin(); l != layers.end(); ++l)
        l->update(rate, b1, b2, t);
}

std::vector<double> DeepAutoencoder::train(Eigen::MatrixXd data, double rate, int epochs)
{
    std::vector<double> out(epochs);

    // Calcualte total data magnitude for normalizing errors
    double data_norm = 0.0;
    for (int i = 0; i < data.cols(); ++i)
    {
        data_norm += data.col(i).norm();
    }

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double e = 0.0;
        for (int i = 0; i < data.cols(); ++i)
        {
            // Encode and decode a datapoint, calculate reconstuction error
            Eigen::VectorXd error = decode(encode(data.col(i))) - data.col(i);
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

std::vector<double> DeepAutoencoder::train(Eigen::MatrixXd data, double rate, int epochs, double b1, double b2)
{
    std::vector<double> out(epochs);

    // Calcualte total data magnitude for normalizing errors
    double data_norm = 0.0;
    for (int i = 0; i < data.cols(); ++i)
    {
        data_norm += data.col(i).norm();
    }

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double e = 0.0;
        for (int i = 0; i < data.cols(); ++i)
        {
            // Encode and decode a datapoint, calculate reconstuction error
            Eigen::VectorXd error = decode(encode(data.col(i))) - data.col(i);
            e += error.norm();

            // Backpropagate errors
            errorLatent(errorReconstruct(error));

            // Update parameters
            update(rate, b1, b2, (epoch * data.cols()) + i + 1);
        }
        out[epoch] = e / data_norm;
    }
    return out;
}