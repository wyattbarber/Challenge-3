#ifndef _POOL2D_HPP
#define _POOL2D_HPP

#include "../Model.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>


namespace neuralnet {

    enum class PoolMode
    {
        Max,
        Min,
        Mean
    };

    template <typename T, int K, PoolMode M>
    class Pool2D : public Model<Pool2D<T, K, M>>
    {
        public:
        typedef Eigen::Tensor<T, 3> InputType;
        typedef Eigen::Tensor<T, 3> OutputType;

        template<typename X>
        OutputType forward(X&& in);

        template<typename X>
        InputType backward(X&& error);

        void update(double rate){}

        Eigen::Tensor<std::pair<int,int>, 3>* get_indices() { return &indices; }

        protected:
        void argcmp(Eigen::Tensor<T,3>& data, int x_start, int x_range, int y_start, int y_range, int channel, T* value);

        Eigen::Tensor<std::pair<int,int>, 3> indices;
        int max_y_idx;
    };


    template <typename T, int K, PoolMode M>
    template<typename X>
    Pool2D<T,K,M>::OutputType Pool2D<T,K,M>::forward(X&& in)
    {
        Eigen::Tensor<T, 3> out(in.dimension(0) / K, in.dimension(1) / K, in.dimension(2));
        indices = Eigen::Tensor<std::pair<int,int>, 3>(in.dimension(0) / K, in.dimension(1) / K, in.dimension(2));
        max_y_idx = in.dimension(0);
        
        Eigen::array<Eigen::Index, 3> out_extent({1, 1, 1});
        Eigen::array<Eigen::Index, 3> pool_dims({1, 1, in.dimension(2)});

        for(int y = 0; y < in.dimension(0); y+=(K+1))
        {
            for(int x = 0; x < in.dimension(1); x+=(K+1))
            {                
                Eigen::array<Eigen::Index, 3> pool_extent({
                    std::min(K, static_cast<int>(in.dimension(0) - y)), 
                    std::min(K, static_cast<int>(in.dimension(1) - x)), 
                    1});
                for(int c = 0; c < in.dimension(2); ++c)
                {                    
                    if constexpr ((M == PoolMode::Max) || (M == PoolMode::Min))
                    {
                        T m;
                        argcmp(in, 
                            x,std::min(K, static_cast<int>(in.dimension(1) - x)), 
                            y,std::min(K, static_cast<int>(in.dimension(0) - y)), 
                            c, &m);
                        out(y/K,x/K,c) = m;
                    } 
                    else if constexpr (M == PoolMode::Mean)
                    {                       
                        Eigen::array<Eigen::Index, 3> pool_start({y, x, c});
                        Eigen::array<Eigen::Index, 3> out_start({y/K, x/K, c});
                        out.slice(out_start, out_extent) = in.slice(pool_start, pool_extent).mean();
                    }
                }
            }   
        }

        return out;
    }


    template <typename T, int K, PoolMode M>
    template<typename X>
    Pool2D<T,K,M>::InputType Pool2D<T,K,M>::backward(X&& error)
    {
        Eigen::Tensor<T, 3> out(error.dimension(0) * K, error.dimension(1) * K, error.dimension(2));
        out.setZero();

        Eigen::array<Eigen::Index, 3> unpool_extent({1, 1, 1});
        Eigen::array<Eigen::Index, 3> out_extent({K, K, 1});
        Eigen::array<Eigen::Index, 3> pool_dims({1, 1, error.dimension(2)});

        for(int y = 0; y < error.dimension(0); ++y)
        {
            for(int x = 0; x < error.dimension(1); ++x)
            {                
                for(int c = 0; c < error.dimension(2); ++c)
                {
                    Eigen::array<Eigen::Index, 3> unpool_start({y, x, c});
                    Eigen::array<Eigen::Index, 3> out_start({y*K, x*K, c});
                    
                    if constexpr ((M == PoolMode::Max) || (M == PoolMode::Min))
                    {
                        auto xo = indices(y,x,c).first;
                        auto yo = indices(y,x,c).second;
                        out(yo, xo, c) = error(y,x,c);
                    }
                    else if constexpr (M == PoolMode::Mean)
                    {
                        for(int xo = 0; xo < K; ++xo)
                        {
                            for(int yo = 0; yo < K; ++yo)
                            {
                                static_cast<Eigen::Tensor<T,3>>(out.slice(out_start, out_extent))(yo,xo,0) = error(y,x,c);                                
                            }
                        }
                    }
                }
            }   
        }

        return out;
    }

    
    template <typename T, int K, PoolMode M>
    void Pool2D<T,K,M>::argcmp(Eigen::Tensor<T,3>& data, int x_start, int x_range, int y_start, int y_range, int channel, T* value)
    {           
        Eigen::array<Eigen::Index, 2> offsets = {y_start, x_start};
        Eigen::array<Eigen::Index, 2> extents = {y_range, x_range};
        auto patch = data.chip(channel,2).slice(offsets,extents);
        Eigen::Tensor<T,0> m;
        if constexpr (M == PoolMode::Max)
        {
            m = patch.maximum();
        }
        else
        {
            m = patch.minimum();
        }
        *value = m(0);

        for(int y = y_start; y <= y_start + y_range; ++y)
        {
            for(int x = x_start; x <= x_start + x_range; ++x)
            {
                if(data(y,x,channel) == *value)
                {
                    indices(y_start/K, x_start/K, channel) = std::make_pair(x,y);
                    break;
                }
            }
        }

        
    }

}
#endif