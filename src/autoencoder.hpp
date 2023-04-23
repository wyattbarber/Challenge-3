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
    Autoencoder(size_t input, size_t latent) : in_size(input), latent_size(latent)
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

    Eigen::VectorXd encode(Eigen::VectorXd input)
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

    Eigen::VectorXd decode(Eigen::VectorXd latent)
    {
        for (int i = 0; i < in_size; ++i)
        {
            arc(i) = latent.dot(W.row(i)) + brc(i);
            if (arc(i) < 0)
                arc(i) = 0.0;
        }
        return arc;
    }

    Eigen::VectorXd errorReconstruct(Eigen::VectorXd error)
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

    Eigen::VectorXd errorLatent(Eigen::VectorXd error)
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

    void update(double rate)
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

    void update(double rate, double b1, double b2, int t)
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

    std::vector<double> train(Eigen::MatrixXd data, double rate, int epochs)
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

    std::vector<double> train(Eigen::MatrixXd data, double rate, int epochs, double b1, double b2)
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
};

class DeepAutoencoder
{
protected:
    std::vector<Autoencoder> layers;

public:
    DeepAutoencoder(std::vector<size_t> dims)
    {
        for (int i = 1; i < dims.size(); ++i)
        {
            layers.push_back(Autoencoder(dims[i - 1], dims[i]));
        }
    }

    Eigen::VectorXd encode(Eigen::VectorXd input)
    {
        std::vector<Eigen::VectorXd> hidden = {input};
        for (int j = 0; j < layers.size(); ++j)
            hidden.push_back(layers[j].encode(hidden.back()));
        return hidden.back();
    }

    Eigen::VectorXd decode(Eigen::VectorXd latent)
    {
        std::vector<Eigen::VectorXd> hidden = {latent};
        for (int j = layers.size() - 1; j >= 0; --j)
            hidden.push_back(layers[j].decode(hidden.back()));
        return hidden.back();
    }

    Eigen::VectorXd errorReconstruct(Eigen::VectorXd error)
    {
        std::vector<Eigen::VectorXd> errors = {error};
        for (int j = 0; j < layers.size(); ++j)
            errors.push_back(layers[j].errorReconstruct(errors.back()));
        return errors.back();
    }

    Eigen::VectorXd errorLatent(Eigen::VectorXd error)
    {
        std::vector<Eigen::VectorXd> errors = {error};
        for (int j = layers.size() - 1; j >= 0; --j)
            errors.push_back(layers[j].errorLatent(errors.back()));
        return errors.back();
    }

    void update(double rate)
    {
        for (auto l = layers.begin(); l != layers.end(); ++l)
        {
            l->update(rate);
        }
    }

    void update(double rate, double b1, double b2, int t)
    {
        for (auto l = layers.begin(); l != layers.end(); ++l)
            l->update(rate, b1, b2, t);
    }

    std::vector<double> train(Eigen::MatrixXd data, double rate, int epochs)
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

    std::vector<double> train(Eigen::MatrixXd data, double rate, int epochs, double b1, double b2)
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
};

class CoupledAutoencoder
{
protected:
    DeepAutoencoder modelA = DeepAutoencoder(std::vector<size_t>(2, 10));
    DeepAutoencoder modelB = DeepAutoencoder(std::vector<size_t>(2, 10));
    size_t latent_size;

public:
    CoupledAutoencoder(std::vector<size_t> dims1, std::vector<size_t> dims2, size_t latent) : latent_size(latent)
    {
        dims1.push_back(latent);
        modelA = DeepAutoencoder(dims1);

        dims2.push_back(latent);
        modelB = DeepAutoencoder(dims2);
    }

    Eigen::VectorXd encode(Eigen::VectorXd input, int domain)
    {
        if(domain == 1) {
            return modelA.encode(input);
        } else {
            return modelB.encode(input);
        }
    }

    Eigen::VectorXd decode(Eigen::VectorXd latent, int domain)
    {
        if(domain == 1){
            return modelA.decode(latent);
        } else {
            return modelB.decode(latent);
        }
    }

    std::vector<std::vector<double>> train(Eigen::MatrixXd dataA, Eigen::MatrixXd dataB, double rate, int epochs, double lambda)
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
                for(int j = 0; j < latent_size; ++j)
                    grA(j) *= (lambda / (embedB(j) + 1e-9));
                modelA.errorLatent(modelA.errorReconstruct(errorA) + grA);

                // Backpropagate model 2
                Eigen::VectorXd grB = (embedB - embedA);
                for(int j = 0; j < latent_size; ++j)
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

    std::vector<std::vector<double>> train(Eigen::MatrixXd dataA, Eigen::MatrixXd dataB, double rate, int epochs, double lambda, double b1, double b2)
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
                for(int j = 0; j < latent_size; ++j)
                    grA(j) *= (lambda / (embedB(j) + 1e-9));
                modelA.errorLatent(modelA.errorReconstruct(errorA) + grA);

                // Backpropagate model 2
                Eigen::VectorXd grB = (embedB - embedA);
                for(int j = 0; j < latent_size; ++j)
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
};
