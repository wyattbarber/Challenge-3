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

        Pool2D(size_t in_x, size_t in_y)
        {
            save_idx = false;
            indices = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>::Zero(in_y, in_x);
        }

        Pool2D(size_t in_x, size_t in_y, Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>& index_dst)
        {
            save_idx = true;
            indices = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>::Zero(in_y, in_x);
            indices_output = &index_dst;
        }

        template<typename X>
        OutputType forward(X&& in);

        template<typename X>
        InputType backward(X&& error);

        void update(double rate){}

        protected:

        bool save_idx;
        Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic> indices;
        Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>* indices_output;
    };


    template <typename T, int K, PoolMode M>
    template<typename X>
    Pool2D<T,K,M>::OutputType Pool2D<T,K,M>::forward(X&& in)
    {
        Eigen::Tensor<T, 3> out(in.dimension(0) / K, in.dimension(1) / K, in.dimension(2));
        out.setZero();
        
        Eigen::array<Eigen::Index, 3> pool_extent({K, K, 1});
        Eigen::array<Eigen::Index, 3> out_extent({1, 1, 1});
        Eigen::array<Eigen::Index, 3> pool_dims({1, 1, in.dimension(2)});

        for(int y = 0; y < in.dimension(0); y+=K)
        {
            for(int x = 0; x < in.dimension(0); x+=K)
            {                
                for(int c = 0; c < in.dimension(2); ++c)
                {
                    Eigen::array<Eigen::Index, 3> pool_start({y, x, c});
                    Eigen::array<Eigen::Index, 3> out_start({y/K, x/K, c});
                    
                    if constexpr (M == PoolMode::Max)
                    {
                        out.slice(out_start, out_extent) = in.slice(pool_start, pool_extent).maximum();
                    }
                    else if constexpr (M == PoolMode::Min)
                    {
                        out.slice(out_start, out_extent) = in.slice(pool_start, pool_extent).minimum();
                    }
                    else if constexpr (M == PoolMode::Mean)
                    {
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

    }
}
#endif