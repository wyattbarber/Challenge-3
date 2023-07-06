#include "include/Variational.hpp"

VariationalAutoencoder::VariationalAutoencoder(std::vector<size_t> dims) : mean{dims[dims.size() - 2], dims[dims.size() - 1]},
                                                                           deviation{dims[dims.size() - 2], dims[dims.size() - 1]},
                                                                           sampler{dims[dims.size() - 1], dims[dims.size() - 2]}
{
    ndist = {0.0, 1.0};

    for (int i = 1; i < dims.size() - 1; ++i)
    {
        layers.push_back(Autoencoder(dims[i - 1], dims[i]));
    }
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> VariationalAutoencoder::encode(Eigen::VectorXd input)
{
    std::vector<Eigen::VectorXd> hidden = {input};
    for (int j = 0; j < layers.size(); ++j)
        hidden.push_back(layers[j].encode(hidden.back()));

    Eigen::VectorXd m = mean.forward(hidden.back());
    Eigen::VectorXd d = deviation.forward(hidden.back());
    return std::make_pair(m, d);
}

Eigen::VectorXd VariationalAutoencoder::decode(Eigen::VectorXd latent)
{
    std::vector<Eigen::VectorXd> hidden = {sampler.forward(latent)};
    for (int j = layers.size() - 1; j >= 0; --j)
        hidden.push_back(layers[j].decode(hidden.back()));
    return hidden.back();
}

Eigen::VectorXd VariationalAutoencoder::errorReconstruct(Eigen::VectorXd error)
{
    std::vector<Eigen::VectorXd> errors = {error};
    for (int j = 0; j < layers.size(); ++j)
        errors.push_back(layers[j].errorReconstruct(errors.back()));
    return sampler.error(errors.back());
}

Eigen::VectorXd VariationalAutoencoder::errorLatent(Eigen::VectorXd error_mean, Eigen::VectorXd error_deviation)
{
    std::vector<Eigen::VectorXd> errors = {mean.error(error_mean) + deviation.error(error_deviation)};
    for (int j = layers.size() - 1; j >= 0; --j)
        errors.push_back(layers[j].errorLatent(errors.back()));
    return errors.back();
}

void VariationalAutoencoder::update(double rate)
{
    for (auto l = layers.begin(); l != layers.end(); ++l)
    {
        l->update(rate);
    }
    mean.update(rate);
    deviation.update(rate);
    sampler.update(rate);
}

void VariationalAutoencoder::update(double rate, double b1, double b2, int t)
{
    for (auto l = layers.begin(); l != layers.end(); ++l)
        l->update(rate, b1, b2, t);
    mean.update(rate, b1, b2, t);
    deviation.update(rate, b1, b2, t);
    sampler.update(rate, b1, b2, t);
}

std::vector<double> VariationalAutoencoder::train(Eigen::MatrixXd data, double rate, int epochs, int L)
{
    std::vector<double> out(epochs);

    // Calculate total data magnitude for normalizing errors
    double data_norm = 0.0;
    for (int i = 0; i < data.cols(); ++i)
    {
        data_norm += data.col(i).norm();
    }

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double avg_loss = 0.0;
        for (int i = 0; i < data.cols(); ++i)
        {
            // Encode a datapoint to generate a distribution
            std::pair<Eigen::VectorXd, Eigen::VectorXd> dist = encode(data.col(i));
            // Calculate KL divergence partial derivatives
            Eigen::VectorXd grad_m = dist.first * -1.0;
            Eigen::VectorXd grad_d = dist.second * -1.0;
            grad_d.unaryExpr([](double x){ return x - (1.0 / (x + DBL_MIN)); });

            // Generate a reconstruction from the sample
            Eigen::VectorXd approx = decode(sample(dist.first, dist.second));

            // Calculate reconstruction error
            Eigen::VectorXd error = approx - data.col(i);

            errorReconstruct(error);
            avg_loss += error.norm();

            // Backpropagate embedding error, penalized by KL divergence
            errorLatent(grad_m, grad_d);

            // Update parameters
            update(rate);
        }
        out[epoch] = avg_loss / data_norm;
    }
    return out;
}

std::vector<double> VariationalAutoencoder::train(Eigen::MatrixXd data, double rate, int epochs, int L, double b1, double b2)
{
    std::vector<double> out(epochs);

    // Calculate total data magnitude for normalizing errors
    double data_norm = 0.0;
    for (int i = 0; i < data.cols(); ++i)
    {
        data_norm += data.col(i).norm();
    }

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double avg_loss = 0.0;
        for (int i = 0; i < data.cols(); ++i)
        {
            // Encode a datapoint to generate a distribution
            std::pair<Eigen::VectorXd, Eigen::VectorXd> dist = encode(data.col(i));
            // Calculate KL divergence partial derivatives
            Eigen::VectorXd grad_m = dist.first * -1.0;
            Eigen::VectorXd grad_d = dist.second * -1.0;
            grad_d.unaryExpr([](double x){ return x - (1.0 / (x + DBL_MIN)); });

            // Generate a reconstruction from the sample
            Eigen::VectorXd approx = decode(sample(dist.first, dist.second));

            // Calculate reconstruction error
            Eigen::VectorXd error = approx - data.col(i);

            errorReconstruct(error);
            avg_loss += error.norm();

            // Backpropagate embedding error, penalized by KL divergence
            errorLatent(grad_m, grad_d);

            // Update parameters
            update(rate, b1, b2, (epoch * data.size()) + i + 1);
        }
        out[epoch] = avg_loss / data_norm;
    }
    return out;
}

Eigen::VectorXd VariationalAutoencoder::sample(Eigen::VectorXd mean, Eigen::VectorXd deviation)
{
    Eigen::VectorXd out = Eigen::VectorXd::Zero(mean.size());

    for (int i = 0; i < mean.size(); ++i)
    {
        double e = ndist(generator);
        out(i) = mean(i) + (deviation(i) * e);
    }

    return out;
}