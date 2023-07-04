#include "../include/Autoencoder.hpp"

Autoencoder::Autoencoder(size_t input, size_t latent) : in_size(input), latent_size(latent)
{
    this->W = Eigen::MatrixXd::Random(input, latent).unaryExpr([input](double x)
                                                               { return x * std::sqrt(2.0 / static_cast<double>(input)); });
    this->blt = Eigen::VectorXd::Zero(latent);
    this->brc = Eigen::VectorXd::Zero(input);
    this->alt = Eigen::VectorXd::Zero(latent);
    this->arc = Eigen::VectorXd::Zero(input);
    this->drc = Eigen::VectorXd::Zero(input);
    this->dlt = Eigen::VectorXd::Zero(latent);

    this->m = Eigen::MatrixXd::Zero(input, latent);
    this->v = Eigen::MatrixXd::Zero(input, latent);
    this->mblt = Eigen::VectorXd::Zero(latent);
    this->mbrc = Eigen::VectorXd::Zero(input);
    this->vblt = Eigen::VectorXd::Zero(latent);
    this->vbrc = Eigen::VectorXd::Zero(input);
}

Eigen::VectorXd Autoencoder::encode(Eigen::VectorXd input)
{
    in = input;
    for (int i = 0; i < latent_size; ++i)
    {
        alt(i) = input.dot(W.col(i)) + blt(i);
        if (alt(i) < 0)
            alt(i) = 0.0;
    }
    return alt;
}

Eigen::VectorXd Autoencoder::decode(Eigen::VectorXd latent)
{
    for (int i = 0; i < in_size; ++i)
    {
        arc(i) = latent.dot(W.row(i)) + brc(i);
        if (arc(i) < 0)
            arc(i) = 0.0;
    }
    return arc;
}

Eigen::VectorXd Autoencoder::errorReconstruct(Eigen::VectorXd error)
{
    for (int i = 0; i < in_size; ++i)
    {
        if (arc(i) > 0)
        {
            drc(i) = error(i);
        }
        else
        {
            drc(i) = 0.0;
        }
    }
    return W.transpose() * drc;
}

Eigen::VectorXd Autoencoder::errorLatent(Eigen::VectorXd error)
{
    for (int i = 0; i < latent_size; ++i)
    {
        if (alt(i) > 0)
        {
            dlt(i) = error(i);
        }
        else
        {
            dlt(i) = 0.0;
        }
    }
    return W * dlt;
}

void Autoencoder::update(double rate)
{
    for (int i = 0; i < in_size; ++i)
    {
        for (int j = 0; j < latent_size; ++j)
        {
            W(i, j) -= rate * ((in(i)*dlt(j)) + (alt(j) * drc(i)));
        }
    }

    blt -= rate * dlt;
    brc -= rate * drc;
}

void Autoencoder::update(double rate, double b1, double b2, int t)
{
    for (int i = 0; i < in_size; ++i)
    {
        for (int j = 0; j < latent_size; ++j)
        {
            double dW = ((in(i)*dlt(j)) + (alt(j) * drc(i)));
            m(i, j) = (b1 * m(i, j)) + ((1.0 - b1) * dW);
            v(i, j) = (b2 * v(i, j)) + ((1.0 - b2) * std::pow(dW, 2.0));

            double mhat = m(i, j) / (1.0 - std::pow(b1, static_cast<double>(t)));
            double vhat = v(i, j) / (1.0 - std::pow(b2, static_cast<double>(t)));

            W(i, j) -= rate * mhat / (std::sqrt(vhat) + 1e-10);
        }
    }
    // Update encoding biases
    mblt = (b1 * mblt) + ((1.0 - b1) * dlt);
    vblt = (b2 * vblt) + ((1.0 - b2) * dlt.cwiseProduct(dlt));

    Eigen::VectorXd mblt_hat = mblt / (1.0 - std::pow(b1, static_cast<double>(t)));
    Eigen::VectorXd vblt_hat = vblt / (1.0 - std::pow(b2, static_cast<double>(t)));
    vblt_hat.unaryExpr([](double x)
                       { return 1.0 / (std::sqrt(x) + 1e-10); });

    blt -= rate * mblt.cwiseProduct(vblt_hat);

    // Update decoding biases
    mbrc = (b1 * mbrc) + ((1.0 - b1) * drc);
    vbrc = (b2 * vbrc) + ((1.0 - b2) * drc.cwiseProduct(drc));

    Eigen::VectorXd mbrc_hat = mbrc / (1.0 - std::pow(b1, static_cast<double>(t)));
    Eigen::VectorXd vbrc_hat = vbrc / (1.0 - std::pow(b2, static_cast<double>(t)));
    vbrc_hat.unaryExpr([](double x)
                       { return 1.0 / (std::sqrt(x) + 1e-10); });

    brc -= rate * mbrc.cwiseProduct(vbrc_hat);
}

std::vector<double> Autoencoder::train(Eigen::MatrixXd data, double rate, int epochs)
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
            Eigen::VectorXd error = decode(encode(data.col(i))) - data.col(i);
            e += error.norm();

            errorLatent(errorReconstruct(error));
            update(rate);
        }
        out[epoch] = e / data_norm;
    }
    return out;
}

std::vector<double> Autoencoder::train(Eigen::MatrixXd data, double rate, int epochs, double b1, double b2)
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
            Eigen::VectorXd error = decode(encode(data.col(i))) - data.col(i);
            e += error.norm();

            errorLatent(errorReconstruct(error));
            update(rate, b1, b2, (epoch * data.cols()) + i + 1);
        }
        out[epoch] = e / data_norm;
    }
    return out;
}