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
     * @param input layer that feeds into this one. Null if this is the first layer.
     */
    Layer(int in_size, int out_size, Layer* input){
        this->weights = Eigen::MatrixXd::Random(in_size, out_size);
        this->biases = Eigen::VectorXd::Zero(out_size);
        this->z = Eigen::VectorXd::Zero(out_size);
        this->a = Eigen::VectorXd::Zero(out_size);
        this->d = Eigen::VectorXd::Zero(out_size);

        this->input_layer = input;
    }

    /**
     * Runs one forward pass through the model
     * 
     * @param input input vector
    */
    Eigen::VectorXd forward(Eigen::VectorXd input){py::print("Base Forward"); return Eigen::VectorXd::Zero(a.size());};
    
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
        std::vector<double> errors;
        py::print("Training called with", inputs.size(), "input data and", outputs.size(), "output data");
        for(int iter = 0; iter < passes; ++iter)
        {
            double e = 0.0;
            for(int i = 0; i < inputs.size(); ++i)
            {
                py::print("Training epoch", iter, "item", i);
                // Test forward pass and calculate error for this input set
                Eigen::VectorXd error = this->forward(inputs.at(i)) - outputs.at(i);
                e += error.array().abs().sum() / inputs.size();

                py::print("Backpropagating");
                this->error(error);
                py::print("Updating parameters");
                this->update(rate);
            }
            errors.push_back(e);
        }
        return errors;
    }

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
    void error(Eigen::VectorXd error){return;};

    /**
     * Updates parameters of this layer based on the previously propagated error
     * @param rate learning rate
    */
    void update(double rate){return;};

    protected:
    Layer* input_layer;

    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    Eigen::VectorXd z;
    Eigen::VectorXd a;
    Eigen::VectorXd d;
};


class ReLU : public Layer {
    public:
    ReLU(int in_size, int out_size, Layer* input) : Layer(in_size, out_size, input){};

    Eigen::VectorXd forward(Eigen::VectorXd input)
    {
        py::print("ReLU Forward Pass");
        if(input_layer == nullptr)
        {   
            z = (input * weights) + biases;
        } else {
            z = (input_layer->forward(input) * weights) + biases;
        }
        a = z.array().max(0).matrix();
        return a;
    }

    protected:
    void error(Eigen::VectorXd error)
    {
        // Calculate this layers error gradient
        d = Eigen::VectorXd::Zero(d.size());
        for(int i = 0; i < z.size(); ++i)
        {
            if(z(i)>0.0) d(i) = 1;
        }

        //backpropagate
        input_layer->error(d);
    }

    void update(double rate)
    {
        for(int n = 0; n < d.size(); ++n) weights.col(n) -= rate * d(n) * a;
        biases -= rate * d;
    }
};


class Sigmoid : public Layer{
    public:
    Sigmoid(int in_size, int out_size, Layer* input) : Layer(in_size, out_size, input){};

    Eigen::VectorXd forward(Eigen::VectorXd input)
    {
        py::print("Sigmoid Forward Pass");
        if(input_layer == nullptr)
        {   
            z = (input * weights) + biases;
        } else {
            z = (input_layer->forward(input) * weights) + biases;
        }
        
        for(int i = 0; i < z.size(); ++i)
        {
            a(i) = 1.0 / (1.0 + std::exp(-z(i)));
        }

        return a;
    };

    protected:
    void error(Eigen::VectorXd error)
    {
        // Calculate this layers error gradient
        d = Eigen::VectorXd::Zero(d.size());
        for(int i = 0; i < d.size(); ++i)
        {
            d(i) = a(i) * (1.0 - a(i));
        }

        //backpropagate
        input_layer->error(d);
    };

    void update(double rate)
    {
        for(int n = 0; n < d.size(); ++n) weights.col(n) -= rate * d(n) * a;
        biases -= rate * d;
    };

};


class Network {
    public:
    Network(std::vector<int> dims, std::vector<activation::ActivationFunc> funcs){
        for(int i = 1; i < dims.size()-2; ++i)
        {
            Layer* h;
            switch(funcs.at(i)){
                case activation::ReLU:
                    *h = ReLU(dims.at(i), dims.at(i+1), &this->layers.back());
                    break;
                case activation::Sigmoid:
                    *h = Sigmoid(dims.at(i), dims.at(i+1), &this->layers.back());
                    break;
                case activation::SoftMax:
                    *h = Sigmoid(dims.at(i), dims.at(i+1), &this->layers.back());
                    break;
            };
            this->layers.push_back(*h);
        }
    }

    Eigen::VectorXd forward(Eigen::VectorXd input){py::print("Network forward pass"); return layers.back().forward(input);}

    std::vector<double> train(std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> outputs, double rate, int passes){
        return layers.back().train(inputs, outputs, rate, passes);
    }

    protected:
    std::vector<Layer> layers;
};

#endif