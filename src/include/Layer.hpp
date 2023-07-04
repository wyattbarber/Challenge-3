#ifndef _LAYER_HPP
#define _LAYER_HPP

#include <Eigen/Dense>
#include <random>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace activation
{
    enum ActivationFunc
    {
        ReLU,
        Sigmoid,
        SoftMax
    };
};

double rnorm()
{
    static std::mt19937 rng;
    static std::normal_distribution<double> nd(0.0, 2.0);
    return nd(rng);
}

class Layer
{

public:
    /**
     * Constructor
     *
     * Constructs a randomly initialized neurual network,
     * where dims.size() specifies the number of layers and dims.at(i)
     * gives the number of nodes in the ith layer. Layer 0 is the input vector,
     * The final layer is the output layer.
     *
     * @param in_size size of the input vector to this layer
     * @param out_size size of the output vector from this layer
     */
    Layer(int in_size, int out_size)
    {
        // Apply che initialization
        this->weights = Eigen::MatrixXd::Random(in_size, out_size).unaryExpr([in_size](double x)
                                                                             { return x * std::sqrt(2.0 / static_cast<double>(in_size)); });

        this->biases = Eigen::VectorXd::Zero(out_size);
        this->z = Eigen::VectorXd::Zero(out_size);
        this->a = Eigen::VectorXd::Zero(out_size);
        this->d = Eigen::VectorXd::Zero(out_size);
        this->in = Eigen::VectorXd::Zero(in_size);

        // Parameters for Adam algorithm
        this->m = Eigen::MatrixXd::Zero(in_size, out_size);
        this->mb = Eigen::VectorXd::Zero(out_size);
        this->v = Eigen::MatrixXd::Zero(in_size, out_size);
        this->vb = Eigen::VectorXd::Zero(out_size);
    }

    /**
     * Runs one forward pass through the model
     *
     * @param input input vector
     * @return output of this layer
     */
    virtual Eigen::VectorXd forward(Eigen::VectorXd input) = 0;

    /**
     * Propagates error over this layer, and back over input layers
     * @param error error gradient of following layer
     * @return error of the input layer to this one
     */
    virtual Eigen::VectorXd error(Eigen::VectorXd error) = 0;

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

    /**
     * Updates parameters of this layer based on the previously propagated error, using the Adam algorithm
     * @param rate learning rate
     * @param b1 first moment decay rate
     * @param b2 second moment decay rate
     * @param t current training step
     */
    void update(double rate, double b1, double b2, int t)
    {
        Eigen::VectorXd mhat, vhat;

        for (int n = 0; n < d.size(); ++n)
        {
            Eigen::VectorXd grad = d(n) * in;
            m.col(n) = (b1 * m.col(n)) + ((1.0 - b1) * grad);
            v.col(n) = (b2 * v.col(n)) + ((1.0 - b2) * (grad.cwiseProduct(grad)));
            mhat = m / (1.0 - std::pow(b1, static_cast<double>(t)));
            vhat = v / (1.0 - std::pow(b2, static_cast<double>(t)));

            for (int i = 0; i < mhat.size(); ++i)
            {
                weights.col(n)(i) -= rate * mhat(i) / std::sqrt(vhat(i) + 0.000001);
            }
        }
        mb = (b1 * mb) + ((1.0 - b1) * d);
        vb = (b2 * vb) + ((1.0 - b2) * (d.cwiseProduct(d)));
        mhat = mb / (1.0 - std::pow(b1, static_cast<double>(t)));
        vhat = vb / (1.0 - std::pow(b2, static_cast<double>(t)));
        for (int i = 0; i < mhat.size(); ++i)
        {
            biases(i) -= rate * mhat(i) / std::sqrt(vhat(i) + 0.000001);
        }
    };

protected:
    Layer *input_layer;

    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    Eigen::VectorXd z;
    Eigen::VectorXd a;
    Eigen::VectorXd d;
    Eigen::VectorXd in;

    Eigen::MatrixXd m, v;
    Eigen::VectorXd mb, vb;
};

#endif