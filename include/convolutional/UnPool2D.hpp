#ifndef _UNPOOL2D_HPP
#define _UNPOOL2D_HPP

#include "Pool2D.hpp"

namespace neuralnet {


    template <typename T, int K, PoolMode M>
    class UnPool2D : public Model<UnPool2D<T, K, M>>
    {
        public:
        typedef Eigen::Tensor<T, 3> InputType;
        typedef Eigen::Tensor<T, 3> OutputType;

        UnPool2D() 
        { 
            static_assert(
                M == PoolMode::Mean, 
                "Min and max unpooling require a reference to a matching pooling layer"
                ); 
        }

        UnPool2D(Pool2D<T,K,M>& pool)
        {
            this->pool = &pool;
        }

        template<typename X>
        OutputType forward(X&& in);

        template<typename X>
        InputType backward(X&& error);

        void update(double rate){}

        protected:
        Pool2D<T,K,M>* pool;
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
                Eigen::array<Eigen::Index, 3> pool_extent({
                    std::min(K, static_cast<int>(in.dimension(0) - y)), 
                    std::min(K, static_cast<int>(in.dimension(1) - x)), 
                    1});
                for(int c = 0; c < in.dimension(2); ++c)
                {                    
                    if constexpr ((M == PoolMode::Max) || (M == PoolMode::Min))
                    {
                        auto xo = (*pool->get_indices())(y,x,c).first;
                        auto yo = (*pool->get_indices())(y,x,c).second;
                        out(yo, xo, c) = in(y,x,c);
                    } 
                    else if constexpr (M == PoolMode::Mean)
                    {    
                        Eigen::array<Eigen::Index, 3> out_extent({K, K, 1});                  
                        Eigen::array<Eigen::Index, 3> out_start({y*K, x*K, c}); 
                        
                        for(int xo = 0; xo < K; ++xo)
                        {
                            for(int yo = 0; yo < K; ++yo)
                            {
                                static_cast<Eigen::Tensor<T,3>>(out.slice(out_start, out_extent))(yo,xo,0) = in(y,x,c);                                
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
        Eigen::array<Eigen::Index, 3> out_extent({1, 1, 1});
        Eigen::array<Eigen::Index, 3> pool_dims({1, 1, error.dimension(2)});

        for(int y = 0; y < out.dimension(0); y+=K)
        {
            for(int x = 0; x < out.dimension(1); x+=K)
            {                
                for(int c = 0; c < out.dimension(2); ++c)
                {                    
                    if constexpr ((M == PoolMode::Max) || (M == PoolMode::Min))
                    {
                        auto xo = (*pool->get_indices())(y,x,c).first;
                        auto yo = (*pool->get_indices())(y,x,c).second;
                        out(y, x, c) = error(yo,xo,c);
                    }
                    else if constexpr (M == PoolMode::Mean)
                    {
                        for(int xo = 0; xo < K; ++xo)
                        {
                            for(int yo = 0; yo < K; ++yo)
                            {
                                Eigen::array<Eigen::Index, 3> pool_start({y*K, x*K, c});
                                Eigen::array<Eigen::Index, 3> out_start({y, x, c});
                                out.slice(out_start, out_extent) = error.slice(pool_start, pool_extent).mean();                                
                            }
                        }
                    }
                }
            }   
        }

        return out;
    }

}
#endif