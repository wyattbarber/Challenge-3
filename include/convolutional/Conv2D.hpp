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
        static_assert((K % 2) == 1, "Kernel size K must be odd.");

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
                // Apply he initialization
                kernels = kernels.setRandom().unaryExpr([](double x)
                            { return x * std::sqrt(2.0 / static_cast<double>(K)); });

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
        
        Eigen::array<std::pair<int, int>, 3> paddings = {
            std::make_pair(K/2, K/2),
            std::make_pair(K/2, K/2),
            std::make_pair(0, 0)
        };
        padded = input.pad(paddings);

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

        Eigen::array<std::pair<int, int>, 3> paddings = {
            std::make_pair(K/2, K/2),
            std::make_pair(K/2, K/2),
            std::make_pair(0, 0)
        };
        Eigen::Tensor<T,3> e_padded = error.pad(paddings);
        
        Eigen::Tensor<T, 3> grad_out(y, x, in_channels);
        grad_out.setZero();

        for(int ko = 0; ko < out_channels; ++ko)
        {
            for(int ki = 0; ki < in_channels; ++ki)
            {
                auto a = padded.chip(ki,2);
                auto kernel = kernels.chip(ko,3).chip(ki,2);
                
                grad_kernels.chip(ko,3).chip(ki,2) = a.convolve(error.chip(ko,2), dims);
                grad_out.chip(ki,2) += e_padded.chip(ko,2).convolve(kernel.reverse(reverse), dims);
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