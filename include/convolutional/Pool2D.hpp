#ifndef _POOL2D_HPP
#define _POOL2D_HPP

#include "../Model.hpp"
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
        
        Pool2D(){ share_indices = false; }
        Pool2D(Eigen::Tensor<int, 3>& x_indices, Eigen::Tensor<int, 3>& y_indices)
        {   
            share_indices = true;
            this->x_indices_shared = &x_indices;
            this->y_indices_shared = &y_indices;
        }
#ifndef NOPYTHON
        Pool2D(const py::tuple&){}
#endif

        template<typename X>
        OutputType forward(X&& in);

        template<typename X>
        InputType backward(X&& error);

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
        void argcmp(Eigen::Tensor<T,3>& data, int x_start, int x_range, int y_start, int y_range, int channel, T* value);

        Eigen::Tensor<int, 3> x_indices, y_indices;
        Eigen::Tensor<int, 3>* x_indices_shared;
        Eigen::Tensor<int, 3>* y_indices_shared;
        bool share_indices;
    };


    template <typename T, int K, PoolMode M>
    template<typename X>
    Pool2D<T,K,M>::OutputType Pool2D<T,K,M>::forward(X&& in)
    {
        Eigen::Tensor<T, 3> out(in.dimension(0) / K, in.dimension(1) / K, in.dimension(2));
        out.setZero();
        if(!share_indices && (in.dimensions() != x_indices.dimensions()))
        {
            x_indices.resize(in.dimensions());
            y_indices.resize(in.dimensions());
        }

        for(int y = 0; y < in.dimension(0); y+=K)
        {
            for(int x = 0; x < in.dimension(1); x+=K)
            {                
                for(int c = 0; c < in.dimension(2); ++c)
                {                    
                    if constexpr ((M == PoolMode::Max) || (M == PoolMode::Min))
                    {
                        T m;
                        argcmp(in, 
                            x,std::min(K, static_cast<int>(in.dimension(1) - x - 1)), 
                            y,std::min(K, static_cast<int>(in.dimension(0) - y - 1)), 
                            c, &m);
                        out(y/K,x/K,c) = m;
                    } 
                    else if constexpr (M == PoolMode::Mean)
                    {                     
                        Eigen::array<Eigen::Index, 3> pool_extent({
                            std::min(K, static_cast<int>(in.dimension(0) - y - 1)), 
                            std::min(K, static_cast<int>(in.dimension(1) - x - 1)), 
                            1}
                        );  
                        Eigen::array<Eigen::Index, 3> pool_start({y, x, c});
                        Eigen::Tensor<T,0> avg = in.slice(pool_start, pool_extent).mean();
                        out(y/K,x/K,c) = avg(0);
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

        auto* x_ind = share_indices ? x_indices_shared : &x_indices;
        auto* y_ind = share_indices ? y_indices_shared : &y_indices;

        for(int y = 0; y < error.dimension(0); ++y)
        {
            for(int x = 0; x < error.dimension(1); ++x)
            {                
                for(int c = 0; c < error.dimension(2); ++c)
                {                    
                    if constexpr ((M == PoolMode::Max) || (M == PoolMode::Min))
                    {
                        auto xo = (*x_ind)(y,x,c);
                        auto yo = (*y_ind)(y,x,c);
                        out(yo, xo, c) = error(y,x,c);
                    }
                    else if constexpr (M == PoolMode::Mean)
                    {
                        for(int xo = 0; xo < K; ++xo)
                        {
                            for(int yo = 0; yo < K; ++yo)
                            {
                                out(y*K + yo, x*K + xo, c) = error(y,x,c);                             
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

        for(int y = y_start; y < y_start + y_range; ++y)
        {
            for(int x = x_start; x < x_start + x_range; ++x)
            {
                if(data(y,x,channel) == m(0))
                {
                    if(share_indices)
                    {
                        (*x_indices_shared)(y_start/K, x_start/K, channel) = x;
                        (*y_indices_shared)(y_start/K, x_start/K, channel) = y;
                    }
                    else
                    {
                        x_indices(y_start/K, x_start/K, channel) = x;
                        y_indices(y_start/K, x_start/K, channel) = y;
                    }
                    break;
                }
            }
        }
    }
}
#endif