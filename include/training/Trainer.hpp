#ifndef _TRAINER_HPP
#define _TRAINER_HPP
#pragma once

#include "../Model.hpp"
#include "../datasource/DataSource.hpp"
#include "../loss/Loss.hpp"
#include <iostream>

using namespace neuralnet;
using namespace datasource;
using namespace loss;

namespace training
{

    /** Handles training of a model on a dataset
     *
     */
    template<class ModelType>
    class Trainer
    {
    public:
        typedef ModelType::Scalar ScalarType;
        typedef ModelType::InputType InputType;
        typedef ModelType::OutputType OutputType;
        typedef DataSource<InputType, OutputType>::SampleType SampleType;
        
        /** Set up a trainer
         *
         * @param model Model to train
         * @param inputs Input dataset for training
         * @param outputs Output dataset for training
         */
        Trainer(ModelType& model, DataSource<InputType, OutputType>& data, Loss<ScalarType>& loss) : 
            model(model),
            data(data),
            loss(loss)
        {
        }

        /** Trains the model over a number of epochs
         * 
         * @param N number of epochs
         * @param rate Learning rate
         * @return Averaged loss for each epoch
        */
        std::vector<ScalarType> train(unsigned N, ScalarType rate);

    protected:
        ModelType& model;
        DataSource<InputType, OutputType>& data;
        Loss<ScalarType>& loss;
    };  
}


template<class ModelType>
std::vector<typename training::Trainer<ModelType>::ScalarType> training::Trainer<ModelType>::train(unsigned N, training::Trainer<ModelType>::ScalarType rate)
{

    std::vector<ScalarType> avg_err;
    
    for (int iter = 0; iter < N; ++iter)
    {
        ScalarType e = 0.0;
        for (int i = 0; i < data.size(); ++i)
        {
            // Test forward pass and calculate error for this input set
            SampleType sample = data.sample(i);
            OutputType out = model.forward(sample.first);

            // Run backward pass
            ScalarType ei;
            OutputType eg = loss.grad(out, sample.second, ei);
            e += ei;
            model.backward(eg);

            // Update model
            model.update(rate);
        }
        avg_err.push_back(e / data.size());
    }
    return avg_err;
}


#endif