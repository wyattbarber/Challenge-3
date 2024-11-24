#ifndef _CONV2D_HPP
#define _CONV2D_HPP

#include "../Model.hpp"
#include "../basic/Activation.hpp"
#include "../optimizers/Optimizer.hpp"
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

            Convolution2D(){ setup(0,0,0,0); }
            Convolution2D(Eigen::Index in_channels, Eigen::Index out_channels, double b1, double b2)
            { setup(in_channels, out_channels, b1, b2); }
            Convolution2D(Eigen::Index in_channels, Eigen::Index out_channels)
            { 
                static_assert(C==OptimizerClass::None, "Adam parameters missing"); 
                setup(in_channels, out_channels); 
            }

            template<typename X>
            OutputType forward(X&& input);

            template<typename X>
            InputType backward(X&& error);

            void update(double rate);
        
#ifndef NOPYTHON
            /** Pickling implementation
             *  
             * @return (in channels, out channels, optimizer args..., kernels, biases)
             */
            static py::tuple getstate(const Convolution2D<T,K,C>& obj);

            static Convolution2D<T,K,C> setstate(py::tuple data);
#endif

        protected:
            Eigen::Index in_channels, out_channels;
            Eigen::Tensor<T,4> kernels;
            Eigen::Tensor<T,1> bias;
            Eigen::Tensor<T,4> grad_kernels;
            Eigen::Tensor<T,1> grad_bias;
            Eigen::Tensor<T,3> padded; // stores input between forward and backward pass

            // Adam optimization data
            adam::AdamData<Eigen::Tensor<T,4>> adam_kernels;
            adam::AdamData<Eigen::Tensor<T,1>> adam_bias;

            template<typename... Ts>
            void setup(Ts... Args)
            {
                auto args = std::tuple<Ts...>(Args...);
                in_channels = std::get<0>(args);
                out_channels = std::get<1>(args);

                kernels = Eigen::Tensor<T,4>(K, K, in_channels, out_channels);
                // Apply he initialization
                kernels = kernels.setRandom().unaryExpr([](double x)
                            { return x * std::sqrt(2.0 / static_cast<double>(K)); });
                bias = Eigen::Tensor<T,1>(out_channels);
                bias.setZero();

                grad_kernels = Eigen::Tensor<T,4>(K, K, in_channels, out_channels);
                grad_bias = Eigen::Tensor<T,1>(out_channels);
                
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

                    adam_bias.m = Eigen::Tensor<T,1>(out_channels);
                    adam_bias.m.setZero();
                    adam_bias.v= Eigen::Tensor<T,1>(out_channels);
                    adam_bias.v.setZero();
                    adam_bias.b1 = std::get<2>(args);
                    adam_bias.b2 = std::get<3>(args);
                    adam_bias.b1powt = adam_bias.b1;
                    adam_bias.b2powt = adam_bias.b2;
                }
            }
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
            out.chip(k,2) = padded.convolve(kernels.chip(k,3), dims).sum(dimsum) + bias(k);
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
            grad_bias(ko) = 0;
            for(int ki = 0; ki < in_channels; ++ki)
            {
                auto a = padded.chip(ki,2);
                auto kernel = kernels.chip(ko,3).chip(ki,2);
                
                grad_kernels.chip(ko,3).chip(ki,2) = a.convolve(error.chip(ko,2), dims);
                Eigen::Tensor<T,0> s = error.chip(ko,2).sum();
                grad_bias(ko) += s(0);
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
            adam::adam_update_params(rate, adam_bias, bias, grad_bias);
        }
        else
        {
            kernels -= rate * grad_kernels;
            bias -= rate * grad_bias;
        }
    }

#ifndef NOPYTHON
    template <typename T, int K, OptimizerClass C>
    py::tuple Convolution2D<T,K,C>::getstate(const Convolution2D<T,K,C>& obj)
    {
        if constexpr (C == OptimizerClass::Adam)
        {
            return py::make_tuple(
                obj.kernels.dimension(2), obj.kernels.dimension(3),
                obj.adam_kernels.b1, obj.adam_kernels.b2,
                std::vector<T>(obj.kernels.data(), obj.kernels.data() + obj.kernels.size()),
                std::vector<T>(obj.bias.data(), obj.bias.data() + obj.bias.size()),
                adam::pickle(obj.adam_kernels),
                adam::pickle(obj.adam_bias)
            );
        }
        else
        {
            return py::make_tuple(
                obj.kernels.dimension(2), obj.kernels.dimension(3),
                std::vector<T>(obj.kernels.data(), obj.kernels.data() + obj.kernels.size()),
                std::vector<T>(obj.bias.data(), obj.bias.data() + obj.bias.size())
            );
        }
    }

    template <typename T, int K, OptimizerClass C>
    Convolution2D<T,K,C> Convolution2D<T,K,C>::setstate(py::tuple data)
    {
        Convolution2D<T,K,C> out;
        std::vector<T> k, b;
        Eigen::Index chn_in = data[0].cast<Eigen::Index>();
        Eigen::Index chn_out = data[1].cast<Eigen::Index>();

        if constexpr (C == OptimizerClass::Adam)
        {
            out = Convolution2D<T,K,C>(chn_in, chn_out, data[2].cast<double>(), data[3].cast<double>());
            k = data[4].cast<std::vector<T>>();
            b = data[5].cast<std::vector<T>>();
            adam::unpickle(data[6], out.adam_kernels);
            adam::unpickle(data[7], out.adam_bias);
        }
        else
        {
            out = Convolution2D<T,K,C>(chn_in, chn_out);
            k = data[4].cast<std::vector<T>>();
            b = data[5].cast<std::vector<T>>();
        }

        auto kmap = Eigen::TensorMap<Eigen::Tensor<T,4>>(k.data(), K,K,chn_in,chn_out);
        out.kernels = kmap;
        
        auto bmap = Eigen::TensorMap<Eigen::Tensor<T,1>>(b.data(), chn_out);
        out.bias = bmap;

        return out;
    }
#endif


}
#endif