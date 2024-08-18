#ifndef _TRAINER_HPP
#define _TRAINER_HPP

#include "../Model.hpp"

using namespace neuralnet;

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
        
        /** Set up a trainer
         *
         * @param model Model to train
         * @param inputs Input dataset for training
         * @param outputs Output dataset for training
         */
        Trainer(ModelType& model, std::vector<InputType>& inputs, std::vector<OutputType>& outputs) : 
            model(model),
            inputs(inputs),
            outputs(outputs)
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
        std::vector<InputType>& inputs; 
        std::vector<OutputType>& outputs;
    };  
}


template<class ModelType>
std::vector<double> training::Trainer<ModelType>::train(unsigned N, double rate)
{
    std::vector<double> avg_err;

    // Total output size for normalizing errors
    double out_norm = 0.0;
    for (int i = 0; i < outputs.size(); ++i)
    {
        out_norm += outputs[i].norm();
    }

    
    for (int iter = 0; iter < N; ++iter)
    {
        double e = 0.0;
        for (int i = 0; i < inputs.size(); ++i)
        {
            // Test forward pass and calculate error for this input set
            InputType in = inputs[i];
            OutputType out = model.forward(in);
            OutputType error = out - outputs[i];
            e += error.norm();
            // Run backward pass
            model.backward(error);
            // Update model
            model.update(rate);
        }
        py::print("Epoch", iter, "average loss:", e / out_norm);
        avg_err.push_back(e / out_norm);
    }
    return avg_err;
}


#endif