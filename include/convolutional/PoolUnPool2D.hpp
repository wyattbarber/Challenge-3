#ifndef _POOL_UNPOOL2D_HPP
#define _POOL_UNPOOL2D_HPP

#include "Pool2D.hpp"
#include "UnPool2D.hpp"

namespace neuralnet {

    /** Paired pool and unpooling layers in an encoder architecture.
     * 
     */
    template <typename T, int K, PoolMode M>
    class PoolUnPool2D : public Encoder<UnPool2D<T, K, M>>
    {
        public:
        typedef Eigen::Tensor<T, 3> InputType;
        typedef Eigen::Tensor<T, 3> OutputType;
        typedef Eigen::Tensor<T, 3> LatentType;

        PoolUnPool2D() : pool(), unpool(pool) {}

        template<typename X>
        OutputType forward(X&& in){ return decode(encode(in)); }

        template<typename X>
        InputType backward(X&& error){ return backward_encode(backward_decode(error)); }

        template<typename X>
        LatentType encode(X&& in){ return pool.forward(in); }

        template<typename X>
        OutputType decode(X&& embed){ return unpool.forward(embed); };

        template<typename X>
        InputType backward_encode(X&& error){return pool.backward(error); }

        template<typename X>
        LatentType backward_decode(X&& error){return unpool.backward(error); }

        void update(double rate){}

#ifndef NOPYTHON
        /** Pickling implementation
         * 
         * This model has no state or constructor args,
         * so it is only defined for compatibility and 
         * to allow it to be part of larger picklable models.
         *  
         * @return empty
         */
        static py::tuple getstate(const PoolUnPool2D<T,K,M>& obj){ return py::tuple(); }

        static PoolUnPool2D<T,K,M> setstate(py::tuple data){ return PoolUnPool2D<T,K,M>(); }
#endif

        protected:

        Pool2D<T,K,M> pool;
        UnPool2D<T,K,M> unpool;
    };
}

#endif