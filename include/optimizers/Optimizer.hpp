#ifndef _OPTIMIZER_HPP
#define _OPTIMIZER_HPP

namespace optimization
{
    /** Basic optimizer interface.
     * 
     * Optimizer types are given to model classes as partially instantiated
     * template parameters, where the Vector, Matrix, or Tensor parameter type 
     * must be instantiated by the "receiving" model.
     * 
     * Each optimizer class should define 2 constructors and 2 methods:
     * 
     * Constructors:
     *  * Unpickling: Takes a py::tuple and restores an optimizer from the encoded state
     *  * Initial: Constructs all data from one or more Eigen::Index values
     * 
     * Methods:
     * * Grad: The `grad` method takes a rate, a reference to parameters, and a gradient of the same type and size,
     *      to then apply an update to the parameters.
     * * Getstate: `getstate` returns a py::tuple encoding the state of the optimizer, for pickling support.
     */
    template<class Derived>
    class Optimizer
    {
        public:
            template<typename T, typename X, typename Y>
            void grad(T rate, X& params, Y&& gradient)
            { return static_cast<Derived*>(this)->grad(rate, params, std::forward<Y>(gradient)); }

#ifndef NOPYTHON
            py::tuple getstate() const { return static_cast<Derived*>(this)->getstate(); }
#endif
    };


    /** Default un-optimized gradient descent 
     * 
     * Implements basic weight update by subtracting the
     * gradient * rate from the current parameter values.
     * 
     * @tparam P multi-dimensional type of the parameter holder
    */
    template< typename P>
    class NoOpt : public Optimizer<NoOpt<P>>
    {
        public:
            typedef P::Scalar Scalar;   

            template<typename... Ts>
            NoOpt(Eigen::Index, Ts...){}
            NoOpt(const py::tuple&){}

            template<typename X>
            void grad(Scalar rate, P& params, X&& gradient){ params -= rate * gradient; }
#ifndef NOPYTHON
            py::tuple getstate() const { return py::tuple(); }
#endif
    };
}

#endif