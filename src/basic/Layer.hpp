#ifndef _LAYER_HPP
#define _LAYER_HPP

#include "../Model.hpp"
#include <random>
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
    template<ActivationFunc F>
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
            // this->weights = Eigen::MatrixXd::Random(in_size, out_size).unaryExpr([in_size](double x)
            //                                                                      { return x * std::sqrt(2.0 / static_cast<double>(in_size)); });
            this->weights = Eigen::MatrixXd::Random(in_size, out_size);

            this->biases = Eigen::VectorXd::Zero(out_size);
            this->z = Eigen::VectorXd::Zero(out_size);
            this->a = Eigen::VectorXd::Zero(out_size);
            this->d = Eigen::VectorXd::Zero(out_size);
            this->in = Eigen::VectorXd::Zero(in_size);

            // // Parameters for Adam algorithm
            // this->m = Eigen::MatrixXd::Zero(in_size, out_size);
            // this->mb = Eigen::VectorXd::Zero(out_size);
            // this->v = Eigen::MatrixXd::Zero(in_size, out_size);
            // this->vb = Eigen::VectorXd::Zero(out_size);
        }

        /**
         * Runs one forward pass through the model
         *
         * @param input input vector
         * @return output of this layer
         */
        Eigen::VectorXd forward(Eigen::VectorXd input) override;

        /**
         * Propagates error over this layer, and back over input layers
         * @param error error gradient of following layer
         * @return error of the input layer to this one
         */
        Eigen::VectorXd backward(Eigen::VectorXd error) override;

        /**
         * Updates parameters of this layer based on the previously propagated error
         * @param rate learning rate
         */
        void update(double rate)
        {
            for (int n = 0; n < d.size(); ++n)
            {
                weights.col(n) -= rate * d(n) * in;
            }
            biases -= rate * d;
        };


    protected:
        Eigen::MatrixXd weights;
        Eigen::VectorXd biases;
        Eigen::VectorXd z;
        Eigen::VectorXd a;
        Eigen::VectorXd d;
        Eigen::VectorXd in;

        // Eigen::MatrixXd m, v;
        // Eigen::VectorXd mb, vb;

        void set_z(Eigen::VectorXd& input)
        {
            in = {input};
            py::print("Data size ", input.size(), ", layer size ", weights.rows(), " x ", weights.cols());
            z = biases;
            py::print("Z set with bias values");
            for(size_t i = 0; i < weights.cols(); ++i)
            {
                py::print(i, "th weighted signal, dot ", in.size(), " with ", weights.col(i).size(), " items");
                double x = in.dot(weights.col(i));
                z(i) += x;
            }
        }
    };

};

#endif