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

#endif