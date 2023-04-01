#ifndef _LAYER_HPP
#define _LAYER_HPP

#include <Eigen/Dense>

#include <pybind11/pybind11.h>
namespace py = pybind11;


namespace activation {
    enum ActivationFunc {
        ReLU,
        Sigmoid,
        SoftMax
    };
};


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
        this->weights = Eigen::MatrixXd::Random(in_size, out_size);
        this->biases = Eigen::VectorXd::Zero(out_size);
        this->z = Eigen::VectorXd::Zero(out_size);
        this->a = Eigen::VectorXd::Zero(out_size);
        this->d = Eigen::VectorXd::Zero(out_size);
        this->in = Eigen::VectorXd::Zero(in_size);
    }

    /**
     * Runs one forward pass through the model
     * 
     * @param input input vector
    */
    virtual Eigen::VectorXd forward(Eigen::VectorXd input) = 0;
    
    
    /** Resets the model to random inital weights
    */
    void reset(){
        weights = Eigen::MatrixXd::Random(weights.rows(), weights.cols());
        biases = Eigen::VectorXd::Zero(biases.size());
        z = Eigen::VectorXd::Zero(z.size());
        a = Eigen::VectorXd::Zero(a.size());
        d = Eigen::VectorXd::Zero(d.size());

        if(input_layer != nullptr)
        {
            input_layer->reset();
        }
    }

    /**
     * Propagates error over this layer, and back over input layers
     * @param error error gradient of following layer
    */
    virtual Eigen::VectorXd error(Eigen::VectorXd error) = 0;

    /**
     * Updates parameters of this layer based on the previously propagated error
     * @param rate learning rate
    */
    virtual void update(double rate) = 0;

    protected:
    Layer* input_layer;

    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    Eigen::VectorXd z;
    Eigen::VectorXd a;
    Eigen::VectorXd d;
    Eigen::VectorXd in;
};


class ReLU : public Layer {
    public:
    ReLU(int in_size, int out_size) : Layer(in_size, out_size){};

    Eigen::VectorXd forward(Eigen::VectorXd input) override
    {
        in = input;
        z = (weights.transpose() * input) + biases;
        py::print("Weighted inputs:");
        for(int j = 0; j < z.size(); ++j)
        {
            py::print('\t', z(j));
        }
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

    void update(double rate) override
    {
        for(int n = 0; n < d.size(); ++n) 
        {
            py::print("Updating node", n, "with gradient", d(n));
            weights.col(n) -= rate * d(n) * in;
        }
        biases -= rate * d;
    }
};


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
            d(i) = err(i) * (a(i) * (1.0 - a(i)));
        }
        return weights * d;
    };

    void update(double rate) override
    {
        for(int n = 0; n < d.size(); ++n) 
        {
            py::print("Updating node", n, "with gradient", d(n));
            weights.col(n) -= rate * d(n) * in;
        }
        biases -= rate * d;
    };

};


class Network {
    public:
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
                    this->layers.at(i-1) = new Sigmoid(dims[i], dims[i-1]);
                    break;
                case activation::SoftMax:
                    this->layers.at(i-1) = new Sigmoid(dims[i], dims[i-1]);
                    break;
            }
        }
    }

    Eigen::VectorXd forward(Eigen::VectorXd input){
        int i = 1;
        std::vector<Eigen::VectorXd> hidden(layers.size()+1);
        hidden[0] = input;

        for(int l = 0; l < layers.size(); ++l)
        {   
            py::print("Input to layer",i,":");
            for(int j = 0; j < hidden[l].size(); ++j)
            {
                py::print('\t', hidden[l](j));
            }
            hidden[l+1] = layers[l]->forward(hidden[l]);
            py::print("Output of layer",i,":");
            for(int j = 0; j < hidden[l+1].size(); ++j)
            {
                py::print('\t', hidden[l+1](j));
            }
            ++i;
        }
        return hidden[layers.size()];
    }

    /**
     * Runs a backpropagation epoch through the model
     * 
     * @param inputs list of N input vectors to train on
     * @param expected list of N correct output vectors
     * @param rate learning rate, default 0.1
     * @param passes number of times to pass over the input data, default 5
     * 
     * @return list containing the error of each test
    */
    std::vector<double> train(std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> outputs, double rate, int passes){
        std::vector<double> avg_err;
        py::print("Training called with", inputs.size(), "input data and", outputs.size(), "output data");
        for(int iter = 0; iter < passes; ++iter)
        {
            double e = 0.0;
            for(int i = 0; i < inputs.size(); ++i)
            {
                py::print("Training epoch", iter, "item", i);

                // Test forward pass and calculate error for this input set
                Eigen::VectorXd errors = this->forward(inputs.at(i)) - outputs.at(i);
                e += errors.array().abs().sum() / double(inputs.size());

                for(int i = layers.size()-1; i >= 0; --i)
                {
                    py::print("Backpropagating");
                    errors = layers[i]->error(errors);
                }

  
                for(auto l = layers.begin(); l != layers.end(); ++l)
                {   
                    py::print("Updating parameters");
                    (*l)->update(rate);
                }
            }
            avg_err.push_back(e);
        }
        return avg_err;
    }


    protected:
    std::vector<Layer*> layers;
};

#endif