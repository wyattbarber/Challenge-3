#ifndef _MODEL_HPP
#define _MODEL_HPP

#include <Eigen/Dense>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#include <utility>

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
    ModelType* makeModel(CTypes... CArgs)
    {
        return new ModelType(CArgs...);
    }
}

#endif
