#ifndef _MODEL_HPP
#define _MODEL_HPP

#include <Eigen/Dense>

namespace neuralnet
{
    /** Abstract class defining basic behavior for neural network components
     *
     * All specific types of layer should inherit from this base class.
     */
    class Model
    {
    public:
        /**
         * Runs one forward pass through the model.
         *
         * @param input input vector
         * @return output of this layer
         */
        virtual Eigen::VectorXd forward(Eigen::VectorXd input) = 0;

        /**
         * Propagates error over this layer, and back over input layers
         *
         * Gradients for this update should be calculated and stored, but
         * parameters not updated. Gradients calculated over multiple calls
         * to this method should be accumulated, to be changed by an update method.
         *
         *
         * @param error error gradient of layer following this one
         * @return error of the layer preceding this one
         */
        virtual Eigen::VectorXd backward(Eigen::VectorXd error) = 0;

        // /** Resets gradients accumulated over previous backward passes to 0.
        //  */
        // virtual void reset() = 0;
        
        /**
         * Updates parameters of this layer based on the previously propagated error
         * @param rate learning rate
         */
        virtual void update(double rate) = 0;

        /**
         * Updates parameters of this layer based on the previously propagated error, using the Adam algorithm
         * @param rate learning rate
         * @param b1 first moment decay rate
         * @param b2 second moment decay rate
         * @param t current training step
         */
        virtual void update(double rate, double b1, double b2, int t) = 0;
    };

}

#endif
