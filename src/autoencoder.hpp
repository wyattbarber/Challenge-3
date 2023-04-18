#include <Eigen/Dense>

class Autoencoder
{
protected:
    size_t in_size, latent_size;

    Eigen::MatrixXd W;
    Eigen::VectorXd blt, brc;
    Eigen::VectorXd in;
    Eigen::VectorXd alt, arc;
    Eigen::VectorXd drc, dlt;

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
        return drc * W.transpose();
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

        // blt -= rate * dlt;
        // brc -= rate * drc;
    }

    std::vector<double> train(Eigen::MatrixXd data, double rate, double epochs)
    {
        std::vector<double> out(epochs);

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            double e = 0.0;
            for (int i = 0; i < data.cols(); ++i)
            {
                Eigen::VectorXd error = decode(encode(data.col(i))) - data.col(i);
                e += error.norm() / (data.col(i).norm() + 1e-15);

                errorLatent(errorReconstruct(error));
                update(rate);
            }
            out[epoch] = e / static_cast<double>(data.cols());
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

    std::vector<double> train(Eigen::MatrixXd data, double rate, double epochs)
    {
        std::vector<double> out(epochs);

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            double e = 0.0;
            for (int i = 0; i < data.cols(); ++i)
            {
                // Encode and decode a datapoint
                std::vector<Eigen::VectorXd> hidden{data.col(i)};
                for (int j = 0; j < layers.size(); ++j)
                    hidden.push_back(layers[j].encode(hidden.back()));
                for (int j = layers.size() - 1; j >= 0; --j)
                    hidden.push_back(layers[j].decode(hidden.back()));

                Eigen::VectorXd error = hidden.back() - data.col(i);
                e += error.norm() / (data.col(i).norm() + 1e-15);

                // Backpropagate decoder error
                std::vector<Eigen::VectorXd> errors{error};
                for (int j = 0; j < layers.size(); ++j)
                    errors.push_back(layers[j].errorReconstruct(errors.back()));
                // Backpropagate encoder error
                for (int j = layers.size() - 1; j >= 0; --j)
                    errors.push_back(layers[j].errorLatent(errors.back()));

                // Update parameters
                for (auto l = layers.begin(); l != layers.end(); ++l)
                    l->update(rate);
            }
            out[epoch] = e / static_cast<double>(data.cols());
        }
        return out;
    }
};
