#ifndef _VALIDATION_HPP
#define _VALIDATION_HPP


#include <pybind11/pybind11.h>
#include <algorithm>
#include <random>
#include "layer.hpp"
namespace py = pybind11;

/** Cross-Validation function
 * 
 * Performs a set of tests on the same model parameters, 
 * assessing percent error based on a percentage of data excluded from testing
 * 
 * @param dims list of layer sizes
 * @param f list of activation functions
 * @param inputs input data vectors for training and testing
 * @param targets list of expected output class for each input
 * @param test_density percentage of datapoints to hold out from training, then test with
 * @param N number of tests to run, each test on a new model and randomized training/test sets
 * @param rate learning rate
 * @param epochs number of trainin epochs for each test
 * 
 * @return list of N error percentages
*/
std::vector<double> test(std::vector<int> dims, std::vector<activation::ActivationFunc> f, std::vector<Eigen::VectorXd> inputs, std::vector<int> targets, double test_density, int N, double rate, int epochs)
{
    std::vector<double> out(N);

    // Vector of indices for selection of training data
    std::vector<int> rand_idx;
    for(int i = 0; i < inputs.size(); ++i) rand_idx.push_back(i);
    // Vector of reordered indices for each test
    std::vector<std::vector<int>> test_indices(N);
    std::random_device rd;
    std::mt19937 g(rd()); // RNG for std::shuffle
    for(int i = 0; i < N; ++i)
    {
        // Randomize training subset
        std::shuffle(rand_idx.begin(), rand_idx.end(), g);
        test_indices[i] = rand_idx;
    }

    int N_points = inputs.size();
    #pragma omp parallel for
    for(int n = 0; n < N; ++n)
    {
        // Create a model
        Network model(dims, f);
        // Select training subset
        int n_test = std::round(static_cast<double>(N_points) * test_density);
        std::vector<Eigen::VectorXd> train_in;
        std::vector<int> train_out;
        for(int i = n_test; i < N_points; ++i)
        {
            train_in.push_back(inputs[test_indices[n][i]]);
            train_out.push_back(targets[test_indices[n][i]]);
        }

        // Train model on new subset
        model.train(train_in, train_out, rate, epochs);

        double error = 0.0;
        // Calculate error on holdout data
        for(int i = 0; i < n_test; ++i)
        {
            Eigen::VectorXd y = model.forward(inputs[test_indices[n][i]]);
            int k_pred = std::distance(y.begin(), std::max_element(y.begin(), y.end()));
            if(k_pred != targets[test_indices[n][i]])
            {
                error += 1.0 / n_test;
            }
        }
        out[n] = error;
    }

    return out;
}


/** Multi-Model Cross-Validation function
 * 
 * Performs a set of tests on a list of model parameters, 
 * assessing percent error based on a percentage of data excluded from testing
 * 
 * @param dims list of lists of layer sizes
 * @param f list of lists of activation functions
 * @param inputs input data vectors for training and testing
 * @param targets list of expected output class for each input
 * @param test_density percentage of datapoints to hold out from training, then test with
 * @param N number of tests to run, each test on a new model and randomized training/test sets
 * @param rate learning rate
 * @param epochs number of trainin epochs for each test
 * 
 * @return size(dims) vectors of N error percentages
*/
std::vector<std::vector<double>> test_dimensions(std::vector<std::vector<int>> dims, std::vector<std::vector<activation::ActivationFunc>> f, std::vector<Eigen::VectorXd> inputs, std::vector<int> targets, double test_density, int N, double rate, int epochs)
{    std::vector<std::vector<double>> results(dims.size());
  
    #pragma omp parallel for
    for(int i = 0; i < dims.size(); ++i) results[i] = test(dims[i], f[i], inputs, targets, test_density, N, rate, epochs);

    return results;
}


#endif