#ifndef _OPTIMIZER_HPP
#define _OPTIMIZER_HPP

#include <Eigen/Dense>

namespace optimization
{
    class Optimizer
    {
        public:
        virtual Optimizer* copy() = 0;

        virtual void init(size_t in_size, size_t out_size) = 0;

        /** Applies optimization algorithm in-place to gradients
         * 
         *
        */
        virtual void augment_gradients(Eigen::MatrixXd& weight_gradients, Eigen::VectorXd& bias_gradients) = 0;

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