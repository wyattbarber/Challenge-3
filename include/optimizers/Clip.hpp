#ifndef _CLIP_HPP
#define _CLIP_HPP

#include "Optimizer.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#ifndef NOPYTHON
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif


namespace optimization
{
    /** Gradient clipping implementation.
     * 
     * @tparam P multidimensional parameter datatype
     * @tparam L limit for clipping
     * @tparam Next nested class to perform subsequent optimization steps.
     */
    template <typename P, P::Scalar L, template<typename> class Next = NoOpt>
    class Clip: public Optimizer<Clip<P,L,Next>>
    {
        public:
            typedef P::Scalar Scalar;   

            template<typename... Ts>
            Clip(Eigen::Index dim0, Ts... Dims) : next(dim0, Dims...) {}
            Clip(const py::tuple& data) : next(data[0]) {}

            template<typename X>
            void grad(Scalar rate, P& params, X&& gradient)
            { 
                next.grad(rate, params, gradient.cwiseMax(-L).cwiseMin(L));
            }
#ifndef NOPYTHON
            py::tuple getstate() const { return py::make_tuple(next.getstate()); }
#endif
        protected:
            Next<P> next;
    };
}

#endif