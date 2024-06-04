#ifndef _MODEL_HPP
#define _MODEL_HPP

#include <Eigen/Dense>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#include <utility>
#include <memory>
#include "Optimizer.hpp"

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
        virtual std::shared_ptr<Eigen::VectorXd> forward(Eigen::VectorXd& input) = 0;

        /**
         * Propagates error over this layer, and back over input layers
         *
         * Error gradients for this pass (or the information needed to
         * calculate them later) should be calculated and stored in this
         * method, but the model parameters should not be updated. 
         * 
         * Gradients calculated over subsequent calls to this method
         * should be accumulated.
         *
         * @param error error gradient of layer following this one
         * @return error of the layer preceding this one
         */
        virtual std::shared_ptr<Eigen::VectorXd> backward(Eigen::VectorXd& error) = 0;

        /**
         * Updates parameters of this layer
         * 
         * Error gradients accumulated over previous calls to backwards are
         * used to update this models weights and biases.
         * 
         * If an optimizer has been defined, it is called here
         * to augment the gradients before applying the update.
         * 
         * @param rate learning rate
         */
        virtual void update(double rate) = 0;

        /** Define an optimization method for the model
         *
         * @param opt Optimizer instance to use
         */
        virtual void apply_optimizer(optimization::Optimizer& opt) = 0;
    };

    /** "Trampoline" class to make pybind11 abstract inheritance work right.
     *
     * @tparam ModelBase derived class, to which virtual method calls will be directed to
     */
    template <class ModelBase>
    class PyModel : public ModelBase
    {
    public:
        using ModelBase::ModelBase; // Inherit constructors
        Eigen::VectorXd forward(Eigen::VectorXd input) override { PYBIND11_OVERRIDE_PURE(Eigen::VectorXd, ModelBase, forward, input); }
        Eigen::VectorXd backward(Eigen::VectorXd error) override { PYBIND11_OVERRIDE(Eigen::VectorXd, ModelBase, backward, error); }
        void update(double rate) override { PYBIND11_OVERRIDE(void, ModelBase, update, rate); }
    };

    /** Factory to create new model instances to pass to python
     *
     * @tparam ModelType concrete model type to create
     * @tparam CArgs concrete model constructor argument types
     */
    template <class ModelType, typename... CTypes>
    ModelType *makeModel(CTypes... CArgs)
    {
        return new ModelType(CArgs...);
    }
}

#endif
