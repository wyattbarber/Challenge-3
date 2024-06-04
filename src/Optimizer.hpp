#ifndef _OPTIMIZER_HPP
#define _OPTIMIZER_HPP

#include <Eigen/Dense>

namespace optimization
{
    class Optimizer
    {
    public:
        /** Initializes the optimizer for a particular model size
         * 
         * Will be called by the model after this optimizer has been assigned to
         * a suitable model instance.
        */
        virtual void init(size_t in_size, size_t out_size) = 0;

        /** Applies optimization algorithm in-place to gradients
         *
         *
         */
        virtual void augment_gradients(Eigen::MatrixXd &weight_gradients, Eigen::VectorXd &bias_gradients) = 0;

        /** Reset all state values of the optimizer to initial values
         *
         */
        virtual void reset() = 0;

        /** Create a new, identical optimizer
         * 
         * Used to propagate an optimizer through composite models.
         * Should copy model parameters, but not states and other model 
         * specific data. The copied optimizer will be assigned to a different model 
         * from this one.
         * 
         * This may be called before or after init has been called, and this 
         * optimizer may be assigned to a model or it may be used as a template
         * and discarded.
         * 
         * @return Pointer to optimizer with identical parameters to this one
         */    
        virtual Optimizer *copy() = 0;

    protected:
        size_t in_size, out_size;
    };

    /** Factory to create new optimizer instances to pass to python
     *
     * @tparam OptType concrete type to create
     * @tparam CArgs concrete constructor argument types
     */
    template <class OptType, typename... CTypes>
    OptType *makeOptimizer(CTypes... CArgs)
    {
        return new OptType(CArgs...);
    }
}

#endif