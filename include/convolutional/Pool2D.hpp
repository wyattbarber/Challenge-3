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

        template<typename X>
        OutputType forward(X&& in);

        template<typename X>
        InputType backward(X&& error);

        void update(double rate){}

        Eigen::Tensor<std::pair<int,int>, 3>* get_indices() { return &indices; }

#ifndef NOPYTHON
        /** Pickling implementation
         * 
         * This model has no state or constructor args,
         * so it is only defined for compatibility and 
         * to allow it to be part of larger picklable models.
         *  
         * @return empty
         */
        static py::tuple getstate(const Pool2D<T,K,M>& obj){ return py::tuple(); }

        static Pool2D<T,K,M> setstate(py::tuple data){ return Pool2D<T,K,M>(); }
#endif

        protected:
        void argcmp(Eigen::Tensor<T,3>& data, int x_start, int x_range, int y_start, int y_range, int channel, T* value);

        Eigen::Tensor<std::pair<int,int>, 3> indices;
    };


    template <typename T, int K, PoolMode M>
    template<typename X>
    Pool2D<T,K,M>::OutputType Pool2D<T,K,M>::forward(X&& in)
    {
        Eigen::Tensor<T, 3> out(in.dimension(0) / K, in.dimension(1) / K, in.dimension(2));
        out.setZero();
        indices = Eigen::Tensor<std::pair<int,int>, 3>(in.dimension(0) / K, in.dimension(1) / K, in.dimension(2));

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

        for(int y = 0; y < error.dimension(0); ++y)
        {
            for(int x = 0; x < error.dimension(1); ++x)
            {                
                for(int c = 0; c < error.dimension(2); ++c)
                {                    
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
                    indices(y_start/K, x_start/K, channel) = std::make_pair(x,y);
                    break;
                }
            }
        }

        
    }
}
#endif