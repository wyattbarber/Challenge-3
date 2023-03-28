#include <stdlib.h>
#include <vector>
#include <Eigen/Dense>

#define VERBOSE

enum ActivationFunc{
    ReLU,
    Sigmoid,
    SoftMax,
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

    /** Resets the model to random inital weights
    */
    void reset();

    protected:
    size_t n_layers;
    std::vector<size_t> dims;
    std::vector<ActivationFunc> function_types;

    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> biases;
    std::vector<Eigen::VectorXd> z;
    std::vector<Eigen::VectorXd> a;
    std::vector<Eigen::VectorXd> d;

    Eigen::VectorXd activation(Eigen::VectorXd, int);
    Eigen::VectorXd d_activation(Eigen::VectorXd, int);

    void err_propagate(Eigen::VectorXd);
    void param_propagate(double);

};


/** Performs a round of cross validation on one model.
 * 
 * Randomly selects training and testing data from the inputs and targets vectors.
 * If the output of the model is size 1, then the error is calculated as the absolute 
 * difference between output and target. If the output size is greater than 1, a mutli-state classifier is assumed,
 * and the error is 1 if the max index matches the max of the target vector, zero otherwise.
 * 
 * @param model NeuralNetwork instance to test
 * @param inputs vector of input data for test and training
 * @param targets vector of correct output data for test and training
 * @param test_density percentage of input data to hold out for testing
 * @param N number of times to repeat the test with new random subsamples of the input data
 * 
 * @return length N vector of doubles, representing the average error of each test iteration
*/
std::vector<double> test(std::vector<size_t> dims, std::vector<ActivationFunc> f, 
std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> targets,
double test_density, int N, double rate, int epochs);