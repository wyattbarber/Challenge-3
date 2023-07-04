#include "../include/CoupledAutoencoder.hpp"

CoupledAutoencoder::CoupledAutoencoder(std::vector<size_t> dims1, std::vector<size_t> dims2, size_t latent) : latent_size(latent)
{
    dims1.push_back(latent);
    modelA = DeepAutoencoder(dims1);

    dims2.push_back(latent);
    modelB = DeepAutoencoder(dims2);
}

Eigen::VectorXd CoupledAutoencoder::encode(Eigen::VectorXd input, int domain)
{
    if (domain == 1)
    {
        return modelA.encode(input);
    }
    else
    {
        return modelB.encode(input);
    }
}

Eigen::VectorXd CoupledAutoencoder::decode(Eigen::VectorXd latent, int domain)
{
    if (domain == 1)
    {
        return modelA.decode(latent);
    }
    else
    {
        return modelB.decode(latent);
    }
}

std::vector<std::vector<double>> CoupledAutoencoder::train(Eigen::MatrixXd dataA, Eigen::MatrixXd dataB, double rate, int epochs, double lambda)
{
    std::vector<std::vector<double>> out(epochs);
    // Calculate total data magnitude for normalizing errors
    double dataA_norm = 0.0;
    double dataB_norm = 0.0;
    for (int i = 0; i < dataA.cols(); ++i)
    {
        dataA_norm += dataA.col(i).norm();
        dataB_norm += dataB.col(i).norm();
    }

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double eA = 0.0;
        double eB = 0.0;
        double eC = 0.0;
        for (int i = 0; i < dataA.cols(); ++i)
        {
            // Encode and decode a datapoint, calculate reconstuction error
            Eigen::VectorXd embedA = encode(dataA.col(i), 1);
            Eigen::VectorXd errorA = decode(embedA, 1) - dataA.col(i);
            eA += errorA.norm();

            Eigen::VectorXd embedB = encode(dataB.col(i), 2);
            Eigen::VectorXd errorB = decode(embedB, 2) - dataB.col(i);
            eB += errorB.norm();

            // Backpropagate model 1
            Eigen::VectorXd grA = (embedA - embedB);
            for (int j = 0; j < latent_size; ++j)
                grA(j) *= (lambda / (embedB(j) + 1e-9));
            modelA.errorLatent(modelA.errorReconstruct(errorA) + grA);

            // Backpropagate model 2
            Eigen::VectorXd grB = (embedB - embedA);
            for (int j = 0; j < latent_size; ++j)
                grB(j) *= (lambda / (embedA(j) + 1e-9));
            modelB.errorLatent(modelB.errorReconstruct(errorB) + grB);

            eC += ((grA / lambda).norm() + (grB / lambda).norm()) / 2.0;

            // Update parameters
            modelA.update(rate);
            modelB.update(rate);
        }
        out[epoch] = {eC / static_cast<double>(dataA.cols()), eA / dataA_norm, eB / dataB_norm};
    }
    return out;
}

std::vector<std::vector<double>> CoupledAutoencoder::train(Eigen::MatrixXd dataA, Eigen::MatrixXd dataB, double rate, int epochs, double lambda, double b1, double b2)
{
    std::vector<std::vector<double>> out(epochs);
    // Calculate total data magnitude for normalizing errors
    double dataA_norm = 0.0;
    double dataB_norm = 0.0;
    for (int i = 0; i < dataA.cols(); ++i)
    {
        dataA_norm += dataA.col(i).norm();
        dataB_norm += dataB.col(i).norm();
    }

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double eA = 0.0;
        double eB = 0.0;
        double eC = 0.0;
        for (int i = 0; i < dataA.cols(); ++i)
        {
            // Encode and decode a datapoint, calculate reconstuction error
            Eigen::VectorXd embedA = encode(dataA.col(i), 1);
            Eigen::VectorXd errorA = decode(embedA, 1) - dataA.col(i);
            eA += errorA.norm();

            Eigen::VectorXd embedB = encode(dataB.col(i), 2);
            Eigen::VectorXd errorB = decode(embedB, 2) - dataB.col(i);
            eB += errorB.norm();

            eC += (embedA - embedB).norm() / ((embedA + embedB).norm() + 1e-9);

            // Backpropagate model 1
            Eigen::VectorXd grA = (embedA - embedB);
            for (int j = 0; j < latent_size; ++j)
                grA(j) *= (lambda / (embedB(j) + 1e-9));
            modelA.errorLatent(modelA.errorReconstruct(errorA) + grA);

            // Backpropagate model 2
            Eigen::VectorXd grB = (embedB - embedA);
            for (int j = 0; j < latent_size; ++j)
                grB(j) *= (lambda / (embedA(j) + 1e-9));
            modelB.errorLatent(modelB.errorReconstruct(errorB) + grB);

            // Update parameters
            modelA.update(rate, b1, b2, (epoch * dataA.cols()) + i + 1);
            modelB.update(rate, b1, b2, (epoch * dataA.cols()) + i + 1);
        }
        out[epoch] = {eC / static_cast<double>(dataA.cols()), eA / dataA_norm, eB / dataB_norm};
    }
    return out;
}