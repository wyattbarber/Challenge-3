#ifndef _POOL2D_HPP
#define _POOL2D_HPP

#include "../Model.hpp"
#include "../basic/Activation.hpp"
#include "../optimizers/Optimizer.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>

#include <pybind11/pybind11.h>
namespace py = pybind11;

using namespace optimization;

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

        Pool2D()
        {
            save_idx = false;
        }

        Pool2D(Eigen::Tensor<Eigen::Index, 3>& index_dst)
        {
            save_idx = true;
            indices_output = &index_dst;
        }

        template<typename X>
        OutputType forward(X&& in);

        template<typename X>
        InputType backward(X&& error);

        void update(double rate){}

        protected:
        void argcmp(Eigen::Tensor<T,3>& data, std::pair<int,int> x_range, std::pair<int,int> y_range, int channel, T& value, Eigen::Index& idx);

        bool save_idx;
        Eigen::Tensor<Eigen::Index, 3> indices;
        int max_y_idx;
        Eigen::Tensor<Eigen::Index, 3>* indices_output;
    };


    template <typename T, int K, PoolMode M>
    template<typename X>
    Pool2D<T,K,M>::OutputType Pool2D<T,K,M>::forward(X&& in)
    {
        Eigen::Tensor<T, 3> out(in.dimension(0) / K, in.dimension(1) / K, in.dimension(2));
        out.setZero();
        indices = Eigen::Tensor<Eigen::Index, 3>(in.dimension(0) / K, in.dimension(1) / K, in.dimension(2));
        max_y_idx = in.dimension(0);
        indices.setZero();
        
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
                            {x,std::min(x+K, static_cast<int>(in.dimension(1) - 1))}, 
                            {y,std::min(y+K, static_cast<int>(in.dimension(0) - 1))}, 
                            c, m, indices(y/K,x/K,c));
                        out(y/K,x/K,c) = m;
                        if(save_idx)
                            (*indices_output)(y/K,x/K,c) = indices(y/K,x/K,c);
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

        for(int y = 0; y < error.dimension(0); y+=K)
        {
            for(int x = 0; x < error.dimension(1); x+=K)
            {                
                for(int c = 0; c < error.dimension(2); ++c)
                {
                    Eigen::array<Eigen::Index, 3> unpool_start({y, x, c});
                    Eigen::array<Eigen::Index, 3> out_start({y*K, x*K, c});
                    
                    if constexpr ((M == PoolMode::Max) || (M == PoolMode::Min))
                    {
                        auto xo = indices(y,x,c) / max_y_idx;
                        auto yo = indices(y,x,c) % max_y_idx;
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
    }

    
    template <typename T, int K, PoolMode M>
    void Pool2D<T,K,M>::argcmp(Eigen::Tensor<T,3>& data, std::pair<int,int> x_range, std::pair<int,int> y_range, int channel, T& value, Eigen::Index& idx)
    {   
        value = data(y_range.first,x_range.first,channel);
        idx = 0;
        for(int y = y_range.first+1; y <= y_range.second; ++y)
        {
            for(int x = x_range.first+1; x <= x_range.second; ++x)
            {
                if constexpr (M == PoolMode::Max)
                {
                    if(data(y,x,channel) > value)
                    {
                        value = data(y,x,channel);
                        idx = (x * data.dimension(0)) + y;
                    }
                }
                else
                {
                    if(data(y,x,channel) < value)
                    {
                        value = data(y,x,channel);
                        idx = (x * data.dimension(0)) + y;
                    }
                }
            }
        }
    }

}
#endif