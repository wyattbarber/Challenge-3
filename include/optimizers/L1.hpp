#ifndef _OPT_L1_HPP
#define _OPT_L1_HPP

#include "Optimizer.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#ifndef NOPYTHON
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif


namespace optimization
{
    /** L1 regularization class
     * 
     * Implemented as a chained optimization class.
     * 
     * @tparam P multidimensional parameter datatype
     * @tparam L regularization coefficient
     * @tparam Next nested class to perform subsequent optimization steps.
     */
    template <typename P, double L, template<typename> class Next = NoOpt>
    class L1 : public Optimizer<L1<P,L,Next>>
    {
        public:
            typedef P::Scalar Scalar;   

            template<typename... Ts>
            L1(Eigen::Index dim0, Ts... Dims) : next(dim0, Dims...) {}
            L1(const py::tuple& data) : next(data[0]) {}

            template<typename X>
            void grad(double rate, P& params, X&& gradient)
            { 
                next.grad(rate, params, gradient + (Scalar(L) * params.unaryExpr([](Scalar x)
                    {
                        return x == Scalar(0) ? 0 : 
                            (
                                x > Scalar(0) ? Scalar(1) : Scalar(-1)
                            );
                    })));
            }
#ifndef NOPYTHON
            py::tuple getstate() const { return py::make_tuple(next.getstate()); }
#endif
        protected:
            Next<P> next;
    };
}

#endif