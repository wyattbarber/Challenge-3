#ifndef _CONV2D_HPP
#define _CONV2D_HPP

#include "../Model.hpp"
#include "../basic/Activation.hpp"
#include "../optimizers/Optimizer.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>

using namespace optimization;

namespace neuralnet {

    template <typename T, int K, OptimizerClass C>
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
                grad_kernels = Eigen::Tensor<T,4>(K, K, in_channels, out_channels);
                
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
            Eigen::Tensor<T, 4> grad_kernels;
            Eigen::Tensor<T,3> padded; // stores input between forward and backward pass

            // Adam optimization data
            adam::AdamData<Eigen::Tensor<T,4>> adam_kernels;
    };


    template<typename T, int K, OptimizerClass C>
    template<typename X>
    Convolution2D<T,K,C>::OutputType Convolution2D<T,K,C>::forward(X&& input)
    {
        const int x = input.dimension(1);
        const int y = input.dimension(0);

        Eigen::Tensor<T,3> out(y, x, out_channels);
        Eigen::array<ptrdiff_t, 2> dims({0, 1});
        Eigen::array<ptrdiff_t, 1> dimsum({2});

        padded = Eigen::Tensor<T,3>(y + K - 1, x + K - 1, in_channels);
        padded.setZero();
        Eigen::array<Eigen::Index, 3> pad_start({K/2, K/2, 0});
        Eigen::array<Eigen::Index, 3> pad_extent({y, x, in_channels});
        padded.slice(pad_start, pad_extent) = input;

        for(int k = 0; k < out_channels; ++k)
        {
            out.chip(k,2) = padded.convolve(kernels.chip(k,3), dims).sum(dimsum);
        }

        return out;
    }


    template<typename T, int K, OptimizerClass C>
    template<typename X>
    Convolution2D<T,K,C>::InputType Convolution2D<T,K,C>::backward(X&& error)
    {   
        const int x = error.dimension(1);
        const int y = error.dimension(0);

        Eigen::array<ptrdiff_t, 2> dims({0, 1});
        Eigen::array<bool, 2> reverse({true, true});
        Eigen::Tensor<T,2> e_padded(y + K - 1, x + K - 1);
        e_padded.setZero();
        Eigen::array<Eigen::Index, 2> pad_start({K/2, K/2});
        Eigen::array<Eigen::Index, 2> pad_extent({y, x});

        Eigen::Tensor<T, 3> grad_out(y, x, in_channels);

        for(int ko = 0; ko < out_channels; ++ko)
        {
            auto e = error.chip(ko,2);
            e_padded.slice(pad_start, pad_extent) = e;

            for(int ki = 0; ki < in_channels; ++ki)
            {
                auto a = padded.chip(ki,2);
                auto kernel = kernels.chip(ko,3).chip(ki,2);
                
                grad_kernels.chip(ko,3).chip(ki,2) = a.convolve(e, dims);
                grad_out.chip(ki,2) += e_padded.convolve(kernel.reverse(reverse), dims);
            }
        }

        return grad_out;
    }


    template<typename T, int K, OptimizerClass C>
    void Convolution2D<T,K,C>::update(double rate)
    {
        if constexpr (C == OptimizerClass::Adam)
        {
            adam::adam_update_params(rate, adam_kernels, kernels, grad_kernels);
        }
        else
        {
            kernels -= rate * grad_kernels;
        }
    }
}
#endif