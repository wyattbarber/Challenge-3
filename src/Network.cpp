#include "include/Network.hpp"

Network::Network(std::vector<int> dims, std::vector<activation::ActivationFunc> funcs)
{
    this->layers = std::vector<Layer *>(dims.size() - 1);
    for (int i = 1; i < dims.size(); ++i)
    {
        switch (funcs.at(i - 1))
        {
        case activation::ReLU:
            this->layers.at(i - 1) = new ReLU(dims[i - 1], dims[i]);
            break;
        case activation::Sigmoid:
            this->layers.at(i - 1) = new Sigmoid(dims[i - 1], dims[i]);
            break;
        case activation::SoftMax:
            this->layers.at(i - 1) = new SoftMax(dims[i - 1], dims[i]);
            break;
        }
    }
}

/**
 * Performs one forward pass, generating output for the complete model.
 *
 * @param input data to pass to the input layer
 * @return output of the final layer
 */
Eigen::VectorXd Network::forward(Eigen::VectorXd input)
{
    std::vector<Eigen::VectorXd> a;
    a.push_back(input);
    for (int l = 0; l < layers.size(); ++l)
    {
        a.push_back(layers[l]->forward(a.back()));
    }
    return a.back();
}

/**
 * Performs backpropagation through the model
 *
 * @param inputs list of N input vectors to train on
 * @param outputs list of N correct output vectors
 * @param rate learning rate
 * @param passes number of times to pass over the input data
 *
 * @return list containing the error of each test
 */
std::vector<double> Network::train(std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> outputs, double rate, int passes)
{
    std::vector<double> avg_err;

    // Total output size for normalizing errors
    double out_norm = 0.0;
    for (int i = 0; i < outputs.size(); ++i)
    {
        out_norm += outputs[i].norm();
    }

    for (int iter = 0; iter < passes; ++iter)
    {
        double e = 0.0;
        for (int i = 0; i < inputs.size(); ++i)
        {
            // Test forward pass and calculate error for this input set
            Eigen::VectorXd error = this->forward(inputs[i]) - outputs[i];

            std::vector<Eigen::VectorXd> errors;
            errors.push_back(error);
            e += error.norm();

            for (int l = layers.size() - 1; l >= 0; --l)
            {
                errors.push_back(layers[l]->error(errors.back()));
                layers[l]->update(rate);
            }
        }
        avg_err.push_back(e / out_norm);
    }
    return avg_err;
}

/**
 * Runs a backpropagation epoch through the model using the Adam algorithm
 *
 * @param inputs list of N input vectors to train on
 * @param outputs list of N correct output vectors
 * @param rate learning rate
 * @param passes number of times to pass over the input data
 *
 * @return list containing the error of each epoch
 */
std::vector<double> Network::train(std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> outputs, double rate, int passes, double b1, double b2)
{
    std::vector<double> avg_err;

    // Total output size for normalizing errors
    double out_norm = 0.0;
    for (int i = 0; i < outputs.size(); ++i)
    {
        out_norm += outputs[i].norm();
    }

    for (int iter = 0; iter < passes; ++iter)
    {
        double e = 0.0;
        for (int i = 0; i < inputs.size(); ++i)
        {
            // Test forward pass and calculate error for this input set
            Eigen::VectorXd error = this->forward(inputs[i]) - outputs[i];

            std::vector<Eigen::VectorXd> errors;
            errors.push_back(error);
            e += error.norm();

            for (int l = layers.size() - 1; l >= 0; --l)
            {
                errors.push_back(layers[l]->error(errors.back()));
                layers[l]->update(rate, b1, b2, (iter * outputs.size()) + i + 1);
            }
        }
        avg_err.push_back(e / out_norm);
    }
    return avg_err;
}
