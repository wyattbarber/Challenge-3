#ifndef _POOL_UNPOOL2D_HPP
#define _POOL_UNPOOL2D_HPP

#include "Pool2D.hpp"
#include "UnPool2D.hpp"

namespace neuralnet {

    /** Paired pool and unpooling layers in an encoder architecture.
     * 
     */
    template <typename T, int K, PoolMode M>
    class PoolUnPool2D : public Encoder<PoolUnPool2D<T, K, M>>
    {
        public:
        typedef Eigen::Tensor<T, 3> InputType;
        typedef Eigen::Tensor<T, 3> OutputType;
        typedef Eigen::Tensor<T, 3> LatentType;

        PoolUnPool2D()
        {
            indices = Eigen::Tensor<std::pair<int,int>, 3>(14,14,3);
            pool = Pool2D<T,K,M>(indices);
            unpool = UnPool2D<T,K,M>(indices);
        }
#ifndef NOPYTHON
        PoolUnPool2D(py::tuple)
        {
            indices = Eigen::Tensor<std::pair<int,int>, 3>(14,14,3);
            pool = Pool2D<T,K,M>(indices);
            unpool = UnPool2D<T,K,M>(indices);
        }
#endif

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
        py::tuple getstate() const { return py::tuple(); }
#endif

        protected:

        Pool2D<T,K,M> pool;
        UnPool2D<T,K,M> unpool;
        Eigen::Tensor<std::pair<int,int>, 3> indices;
    };
}

#endif