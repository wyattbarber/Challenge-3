#ifndef _LAYER_HPP
#define _LAYER_HPP

#include "../Model.hpp"
#include "../Optimizer.hpp"
#include "Activation.hpp"
#include <random>
#include <memory>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace neuralnet
{
    /** Basic layer of a neural network
     *
     * @tparam F Enumerated activation function to use in this layer
     */
    template <int I, int O, typename T, ActivationFunc F>
    class Layer : public Model<I, O, T>
    {

    public:
        /** Constructs a randomly initialized layer.
         *
         * Weights are initialized using He initialization, and
         * biases are initialized to 0.
         *
         * @param in_size size of the input vector to this layer
         * @param out_size size of the output vector from this layer
         */
        Layer(int in_size = I, int out_size = O)
        {
            // // Apply he initialization
            this->weights = Eigen::Matrix<T, I, O>::Random(in_size, out_size).unaryExpr([in_size](double x)
                                                                                        { return x * std::sqrt(2.0 / static_cast<double>(in_size)); });

            this->biases = Eigen::Vector<T, O>::Zero(out_size);
            this->z = Eigen::Vector<T, O>::Zero(out_size);
            this->a = Eigen::Vector<T, O>::Zero(out_size);
            this->d = Eigen::Vector<T, O>::Zero(out_size);
            this->in = Eigen::Vector<T, I>::Zero(in_size);

            this->is_optimized = false;
        }

        std::shared_ptr<Eigen::Vector<T, O>> forward(Eigen::Vector<T, I> &input);

        std::shared_ptr<Eigen::Vector<T, I>> backward(Eigen::Vector<T, O> &error);

        void update(double rate);

        void apply_optimizer(optimization::Optimizer &opt);

    protected:
        Eigen::Matrix<T, I, O> weights;
        Eigen::Vector<T, O> biases;
        Eigen::Vector<T, O> z;
        Eigen::Vector<T, O> a;
        Eigen::Vector<T, O> d;
        Eigen::Vector<T, I> in;

        optimization::Optimizer *opt;
        bool is_optimized;
    };

}


template <int I, int O, typename T, neuralnet::ActivationFunc F>
void neuralnet::Layer<I, O, T, F>::update(double rate)
{
    Eigen::Matrix<T, I, O> weight_grad = in * d.transpose();
    if (is_optimized)
    {
        opt->augment_gradients(weight_grad, d);
    }
    weights -= rate * weight_grad;
    biases -= rate * d;
}

template <int I, int O, typename T, neuralnet::ActivationFunc F>
void neuralnet::Layer<I, O, T, F>::apply_optimizer(optimization::Optimizer &opt)
{
    this->opt = opt.copy();
    if ((I != Eigen::Dynamic) && (O != Eigen::Dynamic))
        this->opt->init(I, O);
    else
        this->opt->init(weights.rows(), weights.cols());
    is_optimized = true;
}

template <int I, int O, typename T, neuralnet::ActivationFunc F>
std::shared_ptr<Eigen::Vector<T, O>> neuralnet::Layer<I, O, T, F>::forward(Eigen::Vector<T, I> &input)
{
    // Save input for this pass and calculate weighted signals
    in = {input};
    z = biases;
    z += weights.transpose() * input;
    // Calculate and save activation function output
    a = neuralnet::activation<F>(z);
    return std::make_shared<Eigen::Vector<T, O>>(a);
}

template <int I, int O, typename T, neuralnet::ActivationFunc F>
std::shared_ptr<Eigen::Vector<T, I>> neuralnet::Layer<I, O, T, F>::backward(Eigen::Vector<T, O> &err)
{
    // Calculate this layers error gradient
    d = neuralnet::d_activation<F>(z, a, err);
    // Calculate and return error gradient input to next layer
    return std::make_shared<Eigen::Vector<T, I>>(weights * d);
}

#endif
