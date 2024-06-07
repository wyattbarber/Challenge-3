#ifndef _TRAINER_HPP
#define _TRAINER_HPP

#include "../Model.hpp"

using namespace neuralnet;

namespace training
{

    /** Handles training of a model on a dataset
     *
     */
    template<int I, int O, typename T>
    class Trainer
    {
    public:
        /** Set up a trainer
         *
         * @param model Model to train
         * @param inputs Input dataset for training
         * @param outputs Output dataset for training
         */
        Trainer(Model<I, O, T>& model, std::vector<Eigen::Vector<T, I>> inputs, std::vector<Eigen::Vector<T, O>> outputs) : model(model)
        {
            this->inputs = inputs;
            this->outputs = outputs;
        }

        /** Trains the model over a number of epochs
         * 
         * @param N number of epochs
         * @param rate Learning rate
         * @return Averaged loss for each epoch
        */
        std::vector<double> train(unsigned N, double rate);

    protected:
        Model<I, O, T> &model;
        std::vector<Eigen::Vector<T, I>> inputs; 
        std::vector<Eigen::Vector<T, O>> outputs;
    };

}



template<int I, int O, typename T>
std::vector<double> training::Trainer<I, O, T>::train(unsigned N, double rate)
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
            Eigen::Vector<T, I> in = inputs[i];
            Eigen::Vector<T, O> out = model.forward(in);
            Eigen::Vector<T, O> error = out - outputs[i];
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