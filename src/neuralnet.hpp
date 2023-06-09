#include <stdlib.h>
#include <vector>
#include <Eigen/Dense>

#define VERBOSE

enum ActivationFunc{
    ReLU,
    Sigmoid,
};

class NeuralNetwork {

    public:
    /**
     * Constructor
     * 
     * Constructs a randomly initialized neurual network, 
     * where dims.size() specifies the number of layers and dims.at(i) 
     * gives the number of nodes in the ith layer. Layer 0 is the input vector, 
     * The final layer is the output layer.
     * 
     * @param dims list of layer sizes
     */
    NeuralNetwork(std::vector<size_t> dims, std::vector<ActivationFunc> f);

    /**
     * Runs one forward pass through the model
     * 
     * @param input input vector
    */
    Eigen::VectorXd forwardPass(Eigen::VectorXd input);
    
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
    std::vector<double> train(std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> outputs, double rate=0.1, int passes=5);


    protected:
    size_t n_layers;
    std::vector<ActivationFunc> function_types;

    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> biases;
    std::vector<Eigen::VectorXd> z;
    std::vector<Eigen::VectorXd> a;
    std::vector<Eigen::VectorXd> d;

    Eigen::VectorXd activation(Eigen::VectorXd, int);
    Eigen::VectorXd d_activation(Eigen::VectorXd, int);
    double activation(double, int);
    double d_activation(double, int);

    void err_propagate(Eigen::VectorXd);
    void param_propagate(double);

};