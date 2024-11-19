#ifndef _TRAINER_HPP
#define _TRAINER_HPP
#pragma once

#include "../Model.hpp"
#include "../datasource/DataSource.hpp"
#include "../loss/L1.hpp"
#include "../loss/L2.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

using namespace neuralnet;
using namespace datasource;

namespace training
{

    /** Handles training of a model on a dataset
     *
     */
    template<class ModelType>
    class Trainer
    {
    public:
        typedef ModelType::InputType InputType;
        typedef ModelType::OutputType OutputType;
        typedef DataSource<InputType, OutputType>::SampleType SampleType;
        
        /** Set up a trainer
         *
         * @param model Model to train
         * @param inputs Input dataset for training
         * @param outputs Output dataset for training
         */
        Trainer(ModelType& model, DataSource<InputType, OutputType>& data) : 
            model(model),
            data(data)
        {
        }

        /** Trains the model over a number of epochs
         * 
         * @param N number of epochs
         * @param rate Learning rate
         * @return Averaged loss for each epoch
        */
        std::vector<double> train(unsigned N, double rate);

    protected:
        ModelType& model;
        DataSource<InputType, OutputType>& data;
        loss::L2<double> loss;
    };  
}


template<class ModelType>
std::vector<double> training::Trainer<ModelType>::train(unsigned N, double rate)
{

    std::vector<double> avg_err;
    
    for (int iter = 0; iter < N; ++iter)
    {
        double e = 0.0;
        for (int i = 0; i < data.size(); ++i)
        {
            // Test forward pass and calculate error for this input set
            SampleType sample = data.sample(i);
            OutputType out = model.forward(sample.first);
            OutputType error = out - sample.second;

            // Run backward pass
            double ei;
            OutputType eg = loss.grad(out, sample.second, ei);
            model.backward(eg);
            // Update model
            model.update(rate);
            e += ei;
        }
        avg_err.push_back(e / data.size());
    }
    return avg_err;
}


#endif