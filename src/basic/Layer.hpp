#ifndef _LAYER_HPP
#define _LAYER_HPP

#include "../Model.hpp"
#include "../Optimizer.hpp"
#include <random>
#include <memory>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace neuralnet
{
    /** Supported activation functions for simple layers
     *
     */
    enum class ActivationFunc
    {
        ReLU,
        Sigmoid,
        SoftMax,
        TanH,
        Linear
    };

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

        void set_z(Eigen::Vector<T, I> &input);
    };

}


#endif
