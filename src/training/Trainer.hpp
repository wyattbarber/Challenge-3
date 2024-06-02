#ifndef _TRAINER_HPP
#define _TRAINER_HPP

#include "../Model.hpp"

using namespace neuralnet;

namespace training
{

    /** Handles training of a model on a dataset
     *
     */
    class Trainer
    {
    public:
        /** Set up a trainer
         *
         * @param model Model to train
         * @param inputs Input dataset for training
         * @param outputs Output dataset for training
         */
        Trainer(Model& model, std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> outputs) : model(model)
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
        Model &model;
        std::vector<Eigen::VectorXd> inputs, outputs;
    };

}

#endif