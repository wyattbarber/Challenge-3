#ifndef _MODEL_HPP
#define _MODEL_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>
#include <memory>
#include <iostream>

#ifndef NOPYTHON
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

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
        template<typename X>
        auto forward(X&& input){return static_cast<ModelType*>(this)->forward(std::forward<X>(input));}

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
        template<typename X>
        auto backward(X&& error)
        {
            if(!train_mode)
            {
                std::cerr << "Cannot perform backward pass in evaluation mode" << std::endl;
                return;
            }
            return static_cast<ModelType*>(this)->backward(std::forward<X>(error));
        }

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
        void update(double rate)
        {
            if(!train_mode)
            {
                std::cerr << "Cannot update in evaluation mode" << std::endl;
                return;
            }
            static_cast<ModelType*>(this)->update(rate);}    

        /** Sets the model to training mode
         * 
         * Enables saving intermediate data for updating model parameters.
         * 
         * If a derived type implements `_mode(bool)`, it will be called with 
         * and argument of true.
         */
        void train()
        {
#ifndef NDEBUG
            std::cout << "Setting training mode" << std::endl;
#endif
            train_mode = true;
            if constexpr(has_mode_func<ModelType>::value)
            {
                static_cast<ModelType*>(this)->_mode(true);
            }

        }

        /** Sets the model to evaluation mode
         * 
         * Disables updating model parameters, and disables storing intermedate data
         * that is unused if only forward passes are performed.
         * 
         * If a derived type implements `_mode(bool)`, it will be called with 
         * and argument of false.
         */
        void eval()
        {
#ifndef NDEBUG
            std::cout << "Setting evaluation mode" << std::endl;
#endif
            train_mode = false;
            if constexpr(has_mode_func<ModelType>::value)
            {
                static_cast<ModelType*>(this)->_mode(false);
            }

        }

#ifndef NOPYTHON
        /** Gets the models current state for pickling.
         * 
         * Should record all the models relevant state info (constructor args
         * and trainable parameters) so that a model could be reconstructed from 
         * the exported info. 
         * 
         * Only info needed to reconstruct the models behavior in forward data
         * processing is needed, intermediate data stored only between forward and 
         * backward passes in training should not be returned.
         * 
         * @return py::tuple containing all necessary state info for the model.
        */
        py::tuple getstate() const { return static_cast<ModelType*>(this)->getstate(); }
#endif
    protected:
        bool train_mode; /// model is in training mode

        template<typename T>
        struct has_mode_func
        {
            template<typename U, void (U::*)()> struct SFINAE {};
            template<typename U> static char Test(SFINAE<U, &U::_mode>*);
            template<typename U> static int Test(...);
            static const bool value = sizeof(Test<T>(0)) == sizeof(char);
        };
    };


    /** Abstract class defining basic behavior for linked encoder type components
     *
     */
    template <class ModelType>
    class Encoder : public Model<ModelType>
    {
        public:

        /** Generate a latent embedding for one input sample
         *
         * @param input input vector
         * @return latent space embedding
         */
        template<typename X>
        auto encode(X&& input){return static_cast<ModelType*>(this)->encode(std::forward<X>(input));}

        /** Decode a latent embedding
         *
         * @param latent embedding vector
         * @return decoded data
         */
        template<typename X>
        auto decode(X&& embed){return static_cast<ModelType*>(this)->decode(std::forward<X>(embed));}

        /** Perform backward pass over the encoder portion
         *
         * @param error error gradient of the embedding
         * @return propagated error gradient
         */
        template<typename X>
        auto backward_encode(X&& error){return static_cast<ModelType*>(this)->backward_encode(std::forward<X>(error));}

        /** Perform backward pass over the decoder portion
         *
         * @param error error gradient of the output
         * @return propagated error gradient
         */
        template<typename X>
        auto backward_decode(X&& error){return static_cast<ModelType*>(this)->backward_decode(std::forward<X>(error));}

    };
}
#endif
