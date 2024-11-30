#ifndef _CONV2D_HPP
#define _CONV2D_HPP

#include "../Model.hpp"
#include "../optimizers/Optimizer.hpp"
#include <algorithm>


using namespace optimization;

namespace neuralnet {

    template <typename T, int K, template<typename> class C>
    class Convolution2D : public Model<Convolution2D<T, K, C>>
    {
        static_assert((K % 2) == 1, "Kernel size K must be odd.");

        public:
            typedef T Scalar;
            typedef Eigen::Tensor<T, 3> InputType;
            typedef Eigen::Tensor<T, 3> OutputType;

            Convolution2D(Eigen::Index in_channels, Eigen::Index out_channels) : 
                kernel_update(K,K,in_channels,out_channels), 
                bias_update(out_channels)
            { setup(in_channels, out_channels); }
#ifndef NOPYTHON
            Convolution2D(const py::tuple& data) : kernel_update(data[4]), bias_update(data[5])
            {
                std::vector<T> k, b;
                Eigen::Index chn_in = data[0].cast<Eigen::Index>();
                Eigen::Index chn_out = data[1].cast<Eigen::Index>();

                setup(chn_in, chn_out);
                k = data[2].cast<std::vector<T>>();
                b = data[3].cast<std::vector<T>>();

                auto kmap = Eigen::TensorMap<Eigen::Tensor<T,4>>(k.data(), K,K,chn_in,chn_out);
                kernels = kmap;
                
                auto bmap = Eigen::TensorMap<Eigen::Tensor<T,1>>(b.data(), chn_out);
                bias = bmap;
            }
#endif

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
            py::tuple getstate() const;
#endif

        protected:
            Eigen::Index in_channels, out_channels;
            Eigen::Tensor<T,4> kernels;
            Eigen::Tensor<T,1> bias;
            Eigen::Tensor<T,4> grad_kernels;
            Eigen::Tensor<T,1> grad_bias;
            Eigen::Tensor<T,3> padded; // stores input between forward and backward pass

            // Adam optimization data
            C<Eigen::Tensor<T,4>> kernel_update;
            C<Eigen::Tensor<T,1>> bias_update;

            void setup(Eigen::Index in_channels, Eigen::Index out_channels)
            {
                this->in_channels = in_channels;
                this->out_channels = out_channels;
#ifndef NDEBUG
                std::cout << "Initializing "<<in_channels<<" to "<<out_channels<<
                    " channel "<<K<<'x'<<K<<" convolution layer" << std::endl;
#endif
                kernels = Eigen::Tensor<T,4>(K, K, in_channels, out_channels);
                // Apply he initialization
                using RNG = Eigen::internal::NormalRandomGenerator<T>;
                kernels = kernels. template setRandom<RNG>().unaryExpr([](T x)
                            { return x * std::sqrt(T(2) / static_cast<T>(K)); });
                bias = Eigen::Tensor<T,1>(out_channels);
                bias.setZero();

                grad_kernels = Eigen::Tensor<T,4>(K, K, in_channels, out_channels);
                grad_bias = Eigen::Tensor<T,1>(out_channels);
            }
    };


    template<typename T, int K, template<typename> class C>
    template<typename X>
    Convolution2D<T,K,C>::OutputType Convolution2D<T,K,C>::forward(X&& input)
    {
        const int x = input.dimension(1);
        const int y = input.dimension(0);
#ifndef NDEBUG
        std::cout << "Convolving "<<y<<'x'<<x<<'x'<<input.dimension(2)<<" tensor" << std::endl;
#endif

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
#ifndef NDEBUG
            std::cout << "Convolving with output filter "<<k << std::endl;
#endif
            out.chip(k,2) = padded.convolve(kernels.chip(k,3), dims).sum(dimsum) + bias(k);
        }
        return out;
    }


    template<typename T, int K, template<typename> class C>
    template<typename X>
    Convolution2D<T,K,C>::InputType Convolution2D<T,K,C>::backward(X&& error)
    {   
#ifndef NDEBUG
        std::cout << "Conv2D backpropagating "<<
            error.dimension(0)<<'x'<<error.dimension(1)<<'x'<<error.dimension(2)<<" gradient" << std::endl;
#endif
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
            grad_bias(ko) = T(0);
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


    template<typename T, int K, template<typename> class C>
    void Convolution2D<T,K,C>::update(double rate)
    {
#ifndef NDEBUG
        std::cout << "Updating convolution layer" << std::endl;
#endif
        kernel_update.grad(rate, kernels, grad_kernels);
        bias_update.grad(rate, bias, grad_bias);
#ifndef NDEBUG
        std::cout << "New average filter magnitude "<<kernels.abs().sum() / static_cast<T>(kernels.size())<< std::endl;
#endif
    }
    

#ifndef NOPYTHON
    template <typename T, int K, template<typename> class C>
    py::tuple Convolution2D<T,K,C>::getstate() const
    {
        return py::make_tuple(
            kernels.dimension(2), kernels.dimension(3),
            std::vector<T>(kernels.data(), kernels.data() + kernels.size()),
            std::vector<T>(bias.data(), bias.data() + bias.size()),
            kernel_update.getstate(),
            bias_update.getstate()
        );
    }
#endif


}
#endif