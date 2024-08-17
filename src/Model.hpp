#ifndef _MODEL_HPP
#define _MODEL_HPP

#include <Eigen/Dense>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#include <utility>
#include <memory>

namespace neuralnet
{
    /** Abstract class defining basic behavior for neural network components
     *
     * All specific types of layer should inherit from this base class.
     */
    template <class ModelType>
    class Model
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
                
        /**
         * Runs one forward pass through the model.
         *
         * @param input input vector
         * @return output of this layer
         */
        auto forward(Eigen::VectorXd &input){return static_cast<ModelType*>(this)->forward(input);}

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
        auto backward(Eigen::VectorXd &error){return static_cast<ModelType*>(this)->backward(error);}

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
        void update(double rate){static_cast<ModelType*>(this)->update(rate);}    
    
    };

    /**
     * 
     */
    template<typename T>
    class DynamicModel
    {
        public:

        typedef Eigen::Vector<T, Eigen::Dynamic> InputType; 
        typedef Eigen::Vector<T, Eigen::Dynamic> OutputType; 

        virtual OutputType forward(InputType &input) = 0;

        virtual InputType backward(OutputType &error) = 0;

        virtual void update(double rate){};
    };

    /**
     * 
     */
    template<typename T, class ModelType>
    class DynamicBinder : public DynamicModel<T>
    {
        public:
        typedef Eigen::Vector<T, Eigen::Dynamic> InputType; 
        typedef Eigen::Vector<T, Eigen::Dynamic> OutputType; 
        
        template<typename... Ts>
        DynamicBinder(Ts... Args) : model(Args...){}


        OutputType forward(InputType &input)
        {
            return model.forward(input);
        }


        InputType backward(OutputType &error)
        {
            return model.backward(error);
        }


        void update(double rate)
        {
            model.update(rate);
        }

        protected:
        ModelType model;
    };

    /** Factory to create new model instances to pass to python
     *
     * @tparam ModelType concrete model type to create
     * @tparam CArgs concrete model constructor argument types
     */
    template <class ModelType, typename... CTypes>
    auto makeModel(CTypes... CArgs)
    {
        return std::shared_ptr<ModelType>(new ModelType(CArgs...));
    };
}

#endif
