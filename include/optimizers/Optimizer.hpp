#ifndef _OPTIMIZER_HPP
#define _OPTIMIZER_HPP

#include "Adam.hpp"

namespace optimization
{
    /** Basic optimizer interface.
     * 
     * Optimizer types are given to model classes as partially instantiated
     * template parameters. The scalar type and parameter type must be instantiated by
     * the "receiving" model.
     * 
     * Each optimizer class should define 2 constructors and 2 methods:
     * 
     * Constructors:
     *  * Unpickling: Takes a py::tuple and restores an optimizer from the encoded state
     *  * Initial: Takes a set of dimensions to initialize the parameter state data for this optimizer
     * 
     * Methods:
     * * Grad: The `grad` method takes a
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
     * @tparam T scalar type of the parameters being optimized
     * @tparam P multi-dimensional type of the parameter holder
    */
    template<typename T, typename P>
    class NoOpt : public Optimizer<NoOpt<T,P>>
    {
        public:
            template<typename... Ts>
            NoOpt(Ts... Args){}
            NoOpt(const py::tuple&){}

            template<typename X>
            void grad(T rate, P& params, X&& gradient){ params -= rate * gradient; }
#ifndef NOPYTHON
            py::tuple getstate() const { return py::tuple(); }
#endif
    };
}

#endif