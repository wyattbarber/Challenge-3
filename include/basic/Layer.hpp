#ifndef _LAYER_HPP
#define _LAYER_HPP

#include "../Model.hpp"
#include "../optimizers/Optimizer.hpp"
#include "Activation.hpp"
#include <random>
#include <memory>

using namespace optimization;

namespace neuralnet
{
    /** Basic layer of a neural network
     *
     * @tparam T Scalar type of the model
     * @tparam F Enumerated activation function to use in this layer
     * @tparam C Optimization function class
     */
    template <typename T, ActivationFunc F, template<typename> class C>
    class Layer : public Model<Layer<T, F, C>>
    {

    public:
        typedef T Scalar;
        typedef Eigen::Vector<T, Eigen::Dynamic> InputType;
        typedef Eigen::Vector<T, Eigen::Dynamic> OutputType;

        /** Constructs a randomly initialized layer.
         *
         * Weights are initialized using He initialization, and
         * biases are initialized to 0.
         *
         * @param in_size size of the input vector to this layer
         * @param out_size size of the output vector from this layer
         */
        Layer(int in_size, int out_size) : 
            weight_update(in_size,out_size), 
            bias_update(out_size) 
        { 
            setup(in_size, out_size); 
        }
#ifndef NOPYTHON
        Layer(const py::tuple& data) : weight_update(data[4]), bias_update(data[5]) 
        {
            std::vector<T> w, b;
            int in = data[0].cast<int>();
            int o = data[1].cast<int>();

            setup(in, o);
            w = data[2].cast<std::vector<T>>();
            b = data[3].cast<std::vector<T>>();

            weights = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(w.data(), in, o);
            biases = Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(b.data(), o);
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
         * @return (in size, out size, optimizer args..., weights, biases, (optimizer state...))
         */
        py::tuple getstate() const;
#endif

    protected:
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> weights;
        OutputType biases;
        OutputType z;
        OutputType a;
        OutputType d;
        InputType in;

        // Optimizers
        C<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>> weight_update;
        C<Eigen::Vector<T,Eigen::Dynamic>> bias_update;

        template <typename... Ts>
        void setup(int in_size, int out_size)
        {
            // Apply he initialization
            this->weights = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Random(in_size, out_size).unaryExpr([in_size](T x)
                                                                                        { return x * std::sqrt(T(2) / static_cast<T>(in_size)); });
            this->biases = OutputType::Zero(out_size);
            this->z = OutputType::Zero(out_size);
            this->a = OutputType::Zero(out_size);
            this->d = OutputType::Zero(out_size);
            this->in = InputType::Zero(in_size);
        }
    };
}

template <typename T, neuralnet::ActivationFunc F, template<typename> class C>
void neuralnet::Layer<T, F, C>::update(double rate)
{
    weight_update.grad(rate, weights, in * d.transpose());
    bias_update.grad(rate, biases, d);
}

template <typename T, neuralnet::ActivationFunc F, template<typename> class C>
template<typename X>
neuralnet::Layer<T, F, C>::OutputType neuralnet::Layer<T, F, C>::forward(X&& input)
{
    // Save input for this pass and calculate weighted signals
    in = {input};
    z = biases;
    z += (weights.transpose() * input);
    // Calculate and save activation function output
    a =  Activation<Eigen::Dynamic, T, F>::f(z);
    return a;
}

template <typename T, neuralnet::ActivationFunc F, template<typename> class C>
template<typename X>
neuralnet::Layer<T, F, C>::InputType neuralnet::Layer<T, F, C>::backward(X&& err)
{
    // Calculate this layers error gradient
    d = Activation<Eigen::Dynamic, T, F>::df(z, a, err);
    // Calculate and return error gradient input to next layer
    return weights * d;
}


#ifndef NOPYTHON
template <typename T, neuralnet::ActivationFunc F, template<typename> class C>
py::tuple neuralnet::Layer<T, F, C>::getstate() const
{
    return py::make_tuple(
        weights.rows(), weights.cols(),
        std::vector<T>(weights.data(), weights.data() + weights.size()),
        std::vector<T>(biases.data(), biases.data() + biases.size()),
        weight_update.getstate(),
        bias_update.getstate()
    );
}
#endif


#endif
