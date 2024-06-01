#include "Sequence.hpp"

/**
 * Performs one forward pass, generating output for the complete model.
 *
 * @param input data to pass to the input layer
 * @return output of the final layer
 */
Eigen::VectorXd neuralnet::Sequence::forward(Eigen::VectorXd input)
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
 * Performs one backward pass through each layer
 *
 * @param err Output error of the model
 * @return Error gradient of the input to the model
 */
Eigen::VectorXd neuralnet::Sequence::backward(Eigen::VectorXd err)
{
    std::vector<Eigen::VectorXd> errors;
    errors.push_back(err);

    for (int l = layers.size() - 1; l >= 0; --l)
    {
        errors.push_back(layers[l]->backward(errors.back()));
    }

    return errors.back();
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
std::vector<double> neuralnet::Sequence::train(std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> outputs, double rate, int passes)
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
            e += error.norm();

            backward(error);
            for (auto l = layers.begin(); l != layers.end(); ++l)
            {
                (*l)->update(rate);
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
std::vector<double> neuralnet::Sequence::train(std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> outputs, double rate, int passes, double b1, double b2)
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
            e += error.norm();

            backward(error);
            for (auto l = layers.begin(); l != layers.end(); ++l)
            {
                (*l)->update(rate, b1, b2, (iter * outputs.size()) + i + 1);
            }
        }
        avg_err.push_back(e / out_norm);
    }
    return avg_err;
}
