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
    template <ActivationFunc F>
    class Layer : public Model
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
        Layer(int in_size, int out_size)
        {
            // // Apply he initialization
            this->weights = Eigen::MatrixXd::Random(in_size, out_size).unaryExpr([in_size](double x)
                                                                                 { return x * std::sqrt(2.0 / static_cast<double>(in_size)); });

            this->biases = Eigen::VectorXd::Zero(out_size);
            this->z = Eigen::VectorXd::Zero(out_size);
            this->a = Eigen::VectorXd::Zero(out_size);
            this->d = Eigen::VectorXd::Zero(out_size);
            this->in = Eigen::VectorXd::Zero(in_size);

            this->is_optimized = false;
        }

        /**
         * Runs one forward pass through the model
         *
         * @param input input vector
         * @return output of this layer
         */
        std::shared_ptr<Eigen::VectorXd> forward(Eigen::VectorXd& input) override;

        /**
         * Propagates error over this layer, and back over input layers
         * @param error error gradient of following layer
         * @return error of the input layer to this one
         */
        std::shared_ptr<Eigen::VectorXd> backward(Eigen::VectorXd& error) override;

        /**
         * Updates parameters of this layer based on the previously propagated error
         * @param rate learning rate
         */
        void update(double rate);

        void apply_optimizer(optimization::Optimizer& opt)
        {
            this->opt = opt.copy();
            this->opt->init(this->weights.rows(), this->weights.cols());
            is_optimized = true;
        }


    protected:
        Eigen::MatrixXd weights;
        Eigen::VectorXd biases;
        Eigen::VectorXd z;
        Eigen::VectorXd a;
        Eigen::VectorXd d;
        Eigen::VectorXd in;
        Eigen::MatrixXd grad_weight;

        optimization::Optimizer* opt;
        bool is_optimized;

        void set_z(Eigen::VectorXd &input);
    };

};

template <neuralnet::ActivationFunc F>
void neuralnet::Layer<F>::update(double rate)
{
    Eigen::MatrixXd weight_grad = in * d.transpose();
    if(is_optimized)
    {
        opt->augment_gradients(weight_grad, d);
    }
    weights -= rate * weight_grad;
    biases -= rate * d;
}

template <neuralnet::ActivationFunc F>
void neuralnet::Layer<F>::set_z(Eigen::VectorXd &input)
{
    in = {input};
    z = biases;
    z += weights.transpose() * input;
}

#endif