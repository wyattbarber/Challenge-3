#include <pybind11/pybind11.h>
#include "neuralnet.hpp"
#include <algorithm>
#include <random>
namespace py = pybind11;


std::vector<double> test(std::vector<size_t> dims, std::vector<ActivationFunc> f, std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> targets, double test_density, int N, double rate, int epochs) 
{
    std::vector<double> out(N);

    // Vector of indices for selection of training data
    std::vector<int> rand_idx;
    for(int i = 0; i < inputs.size(); ++i) rand_idx.push_back(i);

    std::vector<int> targets_argmax;
    for(int i = 0; i < targets.size(); ++i)
    {
        targets_argmax.push_back(std::distance(targets.at(i).begin(), std::max_element(targets.at(i).begin(), targets.at(i).end())));
    }

    #pragma omp parallel for
    for(int n = 0; n < N; ++n)
    {
        // Create a model
        NeuralNetwork model(dims, f);

        // Randomize training subset
        std::random_device rd;
        std::mt19937 g(rd()); // RNG for std::shuffle
        std::shuffle(rand_idx.begin(), rand_idx.end(), g);

        // Select training subset
        int n_test = std::round(rand_idx.size() * test_density);
        std::vector<Eigen::VectorXd> train_in;
        std::vector<Eigen::VectorXd> train_out;
        for(int i = n_test; i < rand_idx.size(); ++i)
        {
            train_in.push_back(inputs.at(i));
            train_out.push_back(targets.at(i));
        }

        // Train model on new subset
        model.train(train_in, train_out, rate, epochs);

        double error = 0.0;
        // Calculate error on holdout data
        for(int i = 0; i < n_test; ++i)
        {
            Eigen::VectorXd y = model.forwardPass(inputs.at(rand_idx.at(i)));
            if(y.size() < 2) 
            {
                error += abs(y(0) - targets.at(rand_idx.at(i))(0)) / n_test;
            } else {
                int k_pred = std::distance(y.begin(), std::max_element(y.begin(), y.end()));
                if(k_pred != targets_argmax.at(rand_idx.at(i)))
                {
                    error += 1.0 / n_test;
                }
            }
        }
        out.at(n) = error;
    }

    return out;
}