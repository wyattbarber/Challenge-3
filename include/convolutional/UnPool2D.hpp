#ifndef _UNPOOL2D_HPP
#define _UNPOOL2D_HPP

#include "Pool2D.hpp"
#include <iostream>


namespace neuralnet {


    template <typename T, int K, PoolMode M>
    class UnPool2D : public Model<UnPool2D<T, K, M>>
    {
        public:
        typedef T Scalar;
        typedef Eigen::Tensor<T, 3> InputType;
        typedef Eigen::Tensor<T, 3> OutputType;


        UnPool2D() {}
        UnPool2D(Eigen::Tensor<int, 3>& x_indices, Eigen::Tensor<int, 3>& y_indices)
        {
            this->x_indices = &x_indices;
            this->y_indices = &y_indices;
        }
#ifndef NOPYTHON
        UnPool2D(const py::tuple&){}
#endif

        template<typename X>
        OutputType forward(X&& in);

        template<typename X>
        InputType backward(X&& error);

        void update(double rate){}

#ifndef NOPYTHON
        /** Pickling implementation
         * 
         * This is only to be used with mean unpooling. Min and
         * max unpooling require a reference to a pooling layer,
         * for pickling compatibility they must be paired in an 
         * encoder class.
         *  
         * @return empty
         */
        py::tuple getstate() const { return py::tuple(); }
#endif

        protected:
        Eigen::Tensor<int, 3>* x_indices;
        Eigen::Tensor<int, 3>* y_indices;
        int max_y_idx;
    };


    template <typename T, int K, PoolMode M>
    template<typename X>
    UnPool2D<T,K,M>::OutputType UnPool2D<T,K,M>::forward(X&& in)
    {
        Eigen::Tensor<T, 3> out(in.dimension(0) * K, in.dimension(1) * K, in.dimension(2));
        out.setZero();
              
        for(int y = 0; y < in.dimension(0); ++y)
        {
            for(int x = 0; x < in.dimension(1); ++x)
            { 
                for(int c = 0; c < in.dimension(2); ++c)
                {                    
                    if constexpr ((M == PoolMode::Max) || (M == PoolMode::Min))
                    {
                        auto xo = (*x_indices)(y,x,c);
                        auto yo = (*y_indices)(y,x,c);
                        out(yo, xo, c) = in(y,x,c); 
                    } 
                    else if constexpr (M == PoolMode::Mean)
                    {                            
                        for(int xo = 0; xo < K; ++xo)
                        {
                            for(int yo = 0; yo < K; ++yo)
                            {
                                out(y*K + yo, x*K + xo, c) = in(y,x,c);                                
                            }
                        }
                    }
                }
            }   
        }

        return out;
    }


    template <typename T, int K, PoolMode M>
    template<typename X>
    UnPool2D<T,K,M>::InputType UnPool2D<T,K,M>::backward(X&& error)
    {
        Eigen::Tensor<T, 3> out(error.dimension(0) / K, error.dimension(1) / K, error.dimension(2));
        out.setZero();

        Eigen::array<Eigen::Index, 3> pool_extent({K, K, 1});

        for(int y = 0; y < out.dimension(0); ++y)
        {
            for(int x = 0; x < out.dimension(1); ++x)
            {                
                for(int c = 0; c < out.dimension(2); ++c)
                {                    
                    if constexpr ((M == PoolMode::Max) || (M == PoolMode::Min))
                    {
                        auto xo = (*x_indices)(y,x,c);
                        auto yo = (*y_indices)(y,x,c);
                        out(y, x, c) = error(yo,xo,c);
                    }
                    else if constexpr (M == PoolMode::Mean)
                    {
                        Eigen::array<Eigen::Index, 3> pool_start({y*K, x*K, c});
                        Eigen::Tensor<T,0> avg = error.slice(pool_start, pool_extent).mean();
                        out(y,x,c) = avg(0);  
                    }
                }
            }   
        }

        return out;
    }

}
#endif