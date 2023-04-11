#ifndef _LAYER_HPP
#define _LAYER_HPP

#include <Eigen/Dense>
#include <random>
#include <pybind11/pybind11.h>
namespace py = pybind11;


namespace activation {
    enum ActivationFunc {
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

class Layer {

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
    Layer(int in_size, int out_size){
        // Apply che initialization
        this->weights = Eigen::MatrixXd::Random(in_size, out_size).unaryExpr([in_size](double x){return x * std::sqrt(2.0 / static_cast<double>(in_size));});
        
        this->biases = Eigen::VectorXd::Zero(out_size);
        this->z = Eigen::VectorXd::Zero(out_size);
        this->a = Eigen::VectorXd::Zero(out_size);
        this->d = Eigen::VectorXd::Zero(out_size);
        this->in = Eigen::VectorXd::Zero(in_size);

        // Parameters for Adam algorithm
        this->m = Eigen::MatrixXd::Zero(in_size , out_size);
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
        for(int n = 0; n < d.size(); ++n) 
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

        for(int n = 0; n < d.size(); ++n) 
        {
            Eigen::VectorXd grad = d(n) * in;
            m.col(n) = (b1 * m.col(n)) + ((1.0 - b1)*grad);
            v.col(n) = (b2 * v.col(n)) + ((1.0 - b2)*(grad.cwiseProduct(grad)));
            mhat = m / (1.0 - std::pow(b1, static_cast<double>(t)));
            vhat = v / (1.0 - std::pow(b2, static_cast<double>(t)));

            for(int i = 0; i < mhat.size(); ++i)
            {
                weights.col(n)(i) -= rate * mhat(i) / std::sqrt(vhat(i) + 0.000001);
            }
        }
        mb = (b1 * mb) + ((1.0 - b1)*d);
        vb = (b2 * vb) + ((1.0 - b2)*(d.cwiseProduct(d)));
        mhat = mb / (1.0 - std::pow(b1, static_cast<double>(t)));
        vhat = vb / (1.0 - std::pow(b2, static_cast<double>(t)));
        for(int i = 0; i < mhat.size(); ++i)
        {
            biases(i) -= rate * mhat(i) / std::sqrt(vhat(i) + 0.000001);
        }
    };


    protected:
    Layer* input_layer;

    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    Eigen::VectorXd z;
    Eigen::VectorXd a;
    Eigen::VectorXd d;
    Eigen::VectorXd in;

    Eigen::MatrixXd m, v;
    Eigen::VectorXd mb, vb;
};

/**
 * Layer that uses ReLU activation function
*/
class ReLU : public Layer {
    public:
    ReLU(int in_size, int out_size) : Layer(in_size, out_size){};

    Eigen::VectorXd forward(Eigen::VectorXd input) override
    {
        in = input;
        z = (weights.transpose() * input) + biases;
        for(int i = 0; i < z.size(); ++i)
        {
            if(z(i) < 0.0)
            {
                a(i) = 0.0;
            } else {
                a(i) = z(i);
            }
        }
        return a;
    }

    protected:
    Eigen::VectorXd error(Eigen::VectorXd err) override
    {
        // Calculate this layers error gradient
        d = Eigen::VectorXd::Zero(d.size());
        for(int i = 0; i < z.size(); ++i)
        {
            if(z(i) > 0.0) 
            {
                d(i) = err(i);
            }
        } 
        return weights * d;
    }
};


/**
 * Layer that uses Sigmoid activation function
*/
class Sigmoid : public Layer{
    public:
    Sigmoid(int in_size, int out_size) : Layer(in_size, out_size){};

    Eigen::VectorXd forward(Eigen::VectorXd input) override
    {
        in = input;
        z = (weights.transpose() * input) + biases;
        
        for(int i = 0; i < z.size(); ++i)
        {
            a(i) = 1.0 / (1.0 + std::exp(-z(i)));
        }

        return a;
    };

    protected:
    Eigen::VectorXd error(Eigen::VectorXd err) override
    {
        // Calculate this layers error gradient
        d = Eigen::VectorXd::Zero(d.size());
        for(int i = 0; i < d.size(); ++i)
        {
            d(i) = err(i) * a(i) * (1.0 - a(i));
        }
        return weights * d;
    };
};


/**
 * Layer that uses SoftMax activation function
*/
class SoftMax : public Layer{
    public:
    SoftMax(int in_size, int out_size) : Layer(in_size, out_size){};

    Eigen::VectorXd forward(Eigen::VectorXd input) override
    {
        in = input;
        z = (weights.transpose() * input) + biases;
        
        for(int i = 0; i < z.size(); ++i)
        {
            a(i) = std::min(std::exp(z(i)), 1e300);
        }

        a /= (a.array().sum() + 0.0000001);

        return a;
    };

    protected:
    Eigen::VectorXd error(Eigen::VectorXd err) override
    {
        // Calculate this layers error gradient
        for(int i = 0; i < d.size(); ++i)
        {
            Eigen::VectorXd kd = Eigen::VectorXd::Zero(d.size());
            kd(i) = 1.0;
            d(i) = err.dot(a(i) * (kd - a));
        }
        return weights * d;
    };
};

/**
 * Network class
 * 
 * This class acts as a container for Layer instances, handling forward and backward propagation through them.
*/
class Network {
    public:
    /** Network constructor
     * 
     * @param dims List of N integers, specifying layer sizes. Size 0 is the input layer, size N-1 is the output layer.
     * @param funcs List of N-1 enumerated activation functions for each hidden layer. Function at N-2 is the output layer.
    */
    Network(std::vector<int> dims, std::vector<activation::ActivationFunc> funcs){
        this->layers = std::vector<Layer*>(dims.size()-1);
        for(int i = 1; i < dims.size(); ++i)
        {
            switch(funcs.at(i-1))
            {
                case activation::ReLU:
                    this->layers.at(i-1) = new ReLU(dims[i-1], dims[i]);
                    break;
                case activation::Sigmoid:
                    this->layers.at(i-1) = new Sigmoid(dims[i-1], dims[i]);
                    break;
                case activation::SoftMax:
                    this->layers.at(i-1) = new SoftMax(dims[i-1], dims[i]);
                    break;
            }
        }
    }

    /**
     * Performs one forward pass, generating output for the complete model.
     * 
     * @param input data to pass to the input layer
     * @return output of the final layer
    */
    Eigen::VectorXd forward(Eigen::VectorXd input){
        std::vector<Eigen::VectorXd> a;
        a.push_back(input);
        for(int l = 0; l < layers.size(); ++l)
        {   
            a.push_back(layers[l]->forward(a.back()));
        }
        return a.back();
    }

    /**
     * Performs backpropagation through the model
     * 
     * @param inputs list of N input vectors to train on
     * @param outputs list of N correct class assignment (range 0 to output size - 1)
     * @param rate learning rate, default 0.1
     * @param passes number of times to pass over the input data, default 5
     * 
     * @return list containing the error of each test
    */
    std::vector<double> train(std::vector<Eigen::VectorXd> inputs, std::vector<int> outputs, double rate, int passes){
        std::vector<double> avg_err;
        for(int iter = 0; iter < passes; ++iter)
        {
            double e = 0.0;
            for(int i = 0; i < inputs.size(); ++i)
            {
                // Test forward pass and calculate error for this input set
                Eigen::VectorXd y = this->forward(inputs[i]);

                Eigen::VectorXd y_exp = Eigen::VectorXd::Zero(y.size());
                y_exp(outputs[i]) = 1.0;

                std::vector<Eigen::VectorXd> errors;
                errors.push_back(y - y_exp);

                if(std::distance(y.begin(), std::max_element(y.begin(), y.end())) != outputs[i])
                {
                    e += 1.0 / static_cast<double>(inputs.size());
                }
                
                for(int l = layers.size()-1; l >= 0; --l)
                {
                    errors.push_back(layers[l]->error(errors.back()));
                    layers[l]->update(rate);
                }
            }
            avg_err.push_back(e);
        }
        return avg_err;
    }


    protected:
    std::vector<Layer*> layers;
};

/**
 * Network that implements the Adam algorithm for training optimization. 
 * Doesn't actually work, generates NaN values in output layer.
*/
class Adam : public Network{
    public:
    Adam(std::vector<int> dims, std::vector<activation::ActivationFunc> funcs) : Network(dims, funcs){}

    /**
     * Runs a backpropagation epoch through the model using the Adam 
     * 
     * @param inputs list of N input vectors to train on
     * @param outputs list of N correct class assignment (range 0 to output size - 1)
     * @param rate learning rate, default 0.1
     * @param passes number of times to pass over the input data, default 5
     * 
     * @return list containing the error of each test
    */
    std::vector<double> train(std::vector<Eigen::VectorXd> inputs, std::vector<int> outputs, double rate, int passes, double b1, double b2) {
        std::vector<double> avg_err;
        int t = 0;
        for(int iter = 0; iter < passes; ++iter)
        {
            double e = 0.0;
            for(int i = 0; i < inputs.size(); ++i)
            {
                // Test forward pass and calculate error for this input set
                Eigen::VectorXd y = this->forward(inputs[i]);

                Eigen::VectorXd y_exp = Eigen::VectorXd::Zero(y.size());
                y_exp(outputs[i]) = 1.0;

                std::vector<Eigen::VectorXd> errors;
                errors.push_back(y - y_exp);

                if(std::distance(y.begin(), std::max_element(y.begin(), y.end())) != outputs[i])
                {
                    e += 1.0 / double(inputs.size());
                }
                
                for(int l = layers.size()-1; l >= 0; --l)
                {
                    errors.push_back(layers[l]->error(errors.back()));
                    layers[l]->update(rate, b1, b2, t);
                }
                ++t;
            }
            avg_err.push_back(e);
        }
        return avg_err;
    }
};

#endif