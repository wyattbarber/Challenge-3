#ifndef _ACTIVATION2D_HPP
#define _ACTIVATION2D_HPP

#include "../Model.hpp"
#include "../basic/Activation.hpp"

using namespace optimization;

namespace neuralnet
{
    /** Basic Layer2D of a neural network
     *
     * @tparam F Enumerated activation function to use in this Layer2D
     */
    template <typename T, ActivationFunc F>
    class Layer2D : public Model<Layer2D<T, F>>
    {

    public:
        typedef Eigen::Tensor<T, 3> InputType;
        typedef Eigen::Tensor<T, 3> OutputType;

#ifndef NOPYTHON
        Layer2D(){}
        Layer2D(py::tuple){}
#endif

        template<typename X>      
        OutputType forward(X&& input);
        
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
        Eigen::Tensor<T, 3> z, a;
    };

    
    /** Applies templated activation functions
     *
     */
    template <typename T, ActivationFunc F>
    class Activation2D
    {
    public:
        template <typename X> static Eigen::Tensor<T, 3> f(X&& input);
        template <typename Y, typename Z> static Eigen::Tensor<T, 3> df(Y&& activation, Z&& error);
    };

    // Specialization definitions
    template<typename T>
    class Activation2D<T, ActivationFunc::Linear>
    {
    public:
        template <typename X> static Eigen::Tensor<T, 3> f(X&& input);
        template <typename Y, typename Z> static Eigen::Tensor<T, 3> df(Y&& activation, Z&& error);
    };

    template<typename T>
    class Activation2D<T, ActivationFunc::ReLU>
    {
    public:
        template <typename X> static Eigen::Tensor<T, 3> f(X&& input);
        template <typename Y, typename Z> static Eigen::Tensor<T, 3> df(Y&& activation, Z&& error);
    };

    template<typename T>
    class Activation2D<T, ActivationFunc::Sigmoid>
    {
    public:
        template <typename X> static Eigen::Tensor<T, 3> f(X&& input);
        template <typename Y, typename Z> static Eigen::Tensor<T, 3> df(Y&& activation, Z&& error);
    };

    template<typename T>
    class Activation2D<T, ActivationFunc::TanH>
    {
    public:
        template <typename X> static Eigen::Tensor<T, 3> f(X&& input);
        template <typename Y, typename Z> static Eigen::Tensor<T, 3> df(Y&& activation, Z&& error);
    };

    template<typename T>
    class Activation2D<T, ActivationFunc::SoftMax>
    {
    public:
        template <typename X> static Eigen::Tensor<T, 3> f(X&& input);
        template <typename Y, typename Z> static Eigen::Tensor<T, 3> df(Y&& activation, Z&& error);
    };
}


template <typename T, neuralnet::ActivationFunc F>
template<typename X>
neuralnet::Layer2D<T, F>::OutputType neuralnet::Layer2D<T, F>::forward(X&& input)
{
    // Calculate and save activation function output
    a =  Activation2D<T, F>::f(input);
    return a;
}

template <typename T, neuralnet::ActivationFunc F>
template<typename X>
neuralnet::Layer2D<T, F>::InputType neuralnet::Layer2D<T, F>::backward(X&& err)
{
    return Activation2D<T, F>::df(a, err);
}


/*
ReLU activation function
*/
template<typename T>
template <typename X> 
Eigen::Tensor<T, 3> neuralnet::Activation2D<T, neuralnet::ActivationFunc::ReLU>::f(X&& input)
{
    return input.cwiseMax(T(0));
}

template<typename T>
template <typename Y, typename Z> 
Eigen::Tensor<T, 3> neuralnet::Activation2D<T, neuralnet::ActivationFunc::ReLU>::df(Y&& activation, Z&& error)
{
    return activation.unaryExpr([](T x){ return x > 0 ? T(1) : T(0); }) * error;
}

/*
Sigmoid activation function
*/
template<typename T>
template <typename X> 
Eigen::Tensor<T, 3> neuralnet::Activation2D<T, neuralnet::ActivationFunc::Sigmoid>::f(X&& input)
{
    return T(1) / (T(1) + (-input).exp());
}

template<typename T>
template <typename Y, typename Z>  
Eigen::Tensor<T, 3> neuralnet::Activation2D<T, neuralnet::ActivationFunc::Sigmoid>::df(Y&& activation, Z&& error)
{
    return error * (activation * (T(1) - activation));
}

/*
TanH activation function
*/
template<typename T>
template <typename X> 
Eigen::Tensor<T, 3> neuralnet::Activation2D<T, neuralnet::ActivationFunc::TanH>::f(X&& input)
{
    auto ex = input.exp();
    auto nex = (-input).exp();
    return (ex - nex) / (ex + nex);
}

template<typename T>
template <typename Y, typename Z> 
Eigen::Tensor<T, 3> neuralnet::Activation2D<T, neuralnet::ActivationFunc::TanH>::df(Y&& activation, Z&& error)
{
    return error * (T(1) - (activation.square()));
}

/*
Softmax activation function
*/
template<typename T>
template <typename X> 
Eigen::Tensor<T, 3> neuralnet::Activation2D<T, neuralnet::ActivationFunc::SoftMax>::f(X&& input)
{
    Eigen::Tensor<T, 3> out(input.dimension(0), input.dimension(1), input.dimension(2));

    for(int y = 0; y < input.dimension(0); ++y)
    {
        for(int x = 0; y < input.dimension(1); ++x)
        {
            auto e = out.chip(y,0).chip(x,1).exp().cwiseMin(1e300); // Prevent exploding values
            Eigen::Tensor<T,0> sum = e.sum();
            T div = abs(sum(0)) < epsilon ? (epsilon * (std::signbit(sum(0)) ? T(-1) : T(1))) : sum(0);
            out.chip(y,0).chip(x,0) = e / div;
        }
    }

    return out;
}

template<typename T>
template <typename Y, typename Z> 
Eigen::Tensor<T, 3> neuralnet::Activation2D<T, neuralnet::ActivationFunc::SoftMax>::df(Y&& activation, Z&& error)
{
    Eigen::Tensor<T, 3> out(error.dimension(0), error.dimension(1), error.dimension(2));
    Eigen::Tensor<T, 1> kd(error.dimension(2));
    kd.setZero();

    for(int y = 0; y < error.dimension(0); ++y)
    {
        for(int x = 0; y < error.dimension(1); ++x)
        {
            for (int c = 0; c < error.dimension(2); ++c)
            {
                kd(c) = T(1); // Kronecker delta, dz_i/dz_j is 1 for i==j, 0 for all others
                Eigen::Tensor<T,1> pixel = activation.chip(y,0).chip(x,1);
                auto dif = kd - pixel;
                auto dot = error.chip(y,0).chip(x,1) * (pixel(c) * dif);
                Eigen::Tensor<T,0> s = dot.sum();
                out(c) = s(0);
                kd(c) = T(0); // Reset dz_i/dz_j
            }
        }
    }

    return out;
}
#endif
