#ifndef _CONV2D_HPP
#define _CONV2D_HPP

#include "../Model.hpp"
#include "../basic/Activation.hpp"
#include "../optimizers/Optimizer.hpp"
#include <unsupported/Eigen/CXX11/Tensor>


using namespace optimization;

namespace neuralnet {

    template <typename T, size_t K, OptimizerClass C>
    class Convolution2D : public Model<Convolution2D<T, K, C>>
    {
        static_assert(std::bool_constant<(K % 2) == 1>::value, "Kernel size K must be odd.");

        public:
            typedef Eigen::Tensor<T, 3> InputType;
            typedef Eigen::Tensor<T, 3> OutputType;

            template <typename... Ts>
            Convolution2D(Ts... Args)
            {        
                auto args = std::tuple<Ts...>(Args...);
                in_channels = std::get<0>(args);
                out_channels = std::get<1>(args);

                kernels = Eigen::Tensor<T,4>(K, K, in_channels, out_channels);
                kernels.setRandom();
                
                if constexpr (C == OptimizerClass::Adam)
                {
                    adam_kernels.m = Eigen::Tensor<T,4>(K, K, in_channels, out_channels);
                    adam_kernels.m.setZero();
                    adam_kernels.v= Eigen::Tensor<T,4>(K, K, in_channels, out_channels);
                    adam_kernels.v.setZero();
                    adam_kernels.b1 = std::get<2>(args);
                    adam_kernels.b2 = std::get<3>(args);
                    adam_kernels.b1powt = adam_kernels.b1;
                    adam_kernels.b2powt = adam_kernels.b2;
                }
            }

            template<typename X>
            OutputType forward(X&& input);

            template<typename X>
            InputType backward(X&& error);

            void update(double rate);
        
        protected:
            Eigen::Index in_channels, out_channels;
            Eigen::Tensor<T,4> kernels;

            // Adam optimization data
            adam::AdamData<Eigen::Tensor<T,4>> adam_kernels;
    };


    template<typename T, size_t K, OptimizerClass C>
    template<typename X>
    Convolution2D<T,K,C>::OutputType Convolution2D<T,K,C>::forward(X&& input)
    {
        auto X = input.dimension(1);
        auto Y = input.dimension(0);
        Eigen::Tensor<T,3> out(Y, X, out_channels);

        Eigen::Array<Eigen::Index, 3, 1> out_offsets = {0, 0, 0};
        Eigen::Array<Eigen::Index, 4, 1> kernel_offsets = {0, 0, 0, 0};
        Eigen::Array<Eigen::Index, 3, 1> out_extents = {Y,X,1};
        Eigen::Array<Eigen::Index, 4, 1> kernel_extents = {K, K, in_channels, 1};

        for(size_t i = 0; i < out_channels; ++i)
        {
            out_offsets(3,0) = i;
            kernel_offsets(4,0) = i;
            Eigen::array<ptrdiff_t, 3> dims = {0,1,2};
            out.chip(i, 3) = static_cast<InputType>(input).convolve(kernels.chip(i, 4), dims);
        }
        return out;
    }


    template<typename T, size_t K, OptimizerClass C>
    template<typename X>
    Convolution2D<T,K,C>::InputType Convolution2D<T,K,C>::backward(X&& error)
    {

    }
}
#endif