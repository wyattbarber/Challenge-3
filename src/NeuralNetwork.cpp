#include <pybind11/pybind11.h>
#include "neuralnet.hpp"
namespace py = pybind11;

NeuralNetwork::NeuralNetwork(std::vector<size_t> dims)
{
   this->n_layers = dims.size() - 1;

   for(size_t i = 1; i <= n_layers; ++i){
      Eigen::MatrixXd w_i = Eigen::MatrixXd::Random(dims.at(i-1), dims.at(i));
      this->weights.push_back(w_i);

      Eigen::VectorXd b_i = Eigen::VectorXd::Zero(dims.at(i));
      this->biases.push_back(b_i);

      Eigen::VectorXd h_i = Eigen::VectorXd::Zero(dims.at(i));
      this->z.push_back(h_i);
      this->a.push_back(h_i);
   }
}

Eigen::VectorXd NeuralNetwork::forwardPass(Eigen::VectorXd input)
{
#ifdef VERBOSE
   py::print("Calling forward pass...");
#endif

   z.at(0) = input * weights.at(0);
   a.at(0) = this->activation(z.at(0));
   for(size_t i = 1; i < n_layers; ++i){
      z.at(i) = a.at(i-1) * weights.at(i);
      a.at(i) = this->activation(z.at(i));
   }
   return a.at(n_layers-1);
}

std::vector<double>* NeuralNetwork::backprop(std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> outputs, double rate, int passes)
{
#ifdef VERBOSE
   py::print("Calling back propagation...");
#endif

   std::vector<double> errors;
   errors.reserve(passes * inputs.size());

   for(int iter = 0; iter < passes; ++iter)
   {
      for(int i = 0; i < inputs.size(); ++i)
      {
         // Test forward pass and calculate error for this input set
#ifdef VERBOSE
         py::print("Calling forward pass and calculating error");
#endif
         Eigen::VectorXd output = this->forwardPass(inputs.at(i));
         double error = abs((output - outputs.at(i)).array()).sum();
         errors.push_back(error);

         // Iterate over layers
         for(int j = n_layers-1; j >= 0 ; --j)
         {
#ifdef VERBOSE
            py::print("Backpropagating over layer ", j, " of ", n_layers);
#endif
            // Iterate over all weights and nodes in this layer
            #pragma omp parallel for
            for(int k = 0; k < weights.at(j).size(); ++k)
            {
               size_t col = std::floor(i / weights.at(j).cols()); // Index of this node
               size_t row = i % weights.at(j).cols(); // Index of this weight

               // Calculate gradient at this weight
               double grad;
               if(j > 0){  
                  // Hidden or output layer
                  grad = weights.at(j)(row, col) * error * this->d_activation(z.at(j-1)(row));
                  weights.at(j)(row, col) += rate * grad * a.at(j-1)(row);
               } else {
                  // Input layer
                  grad = weights.at(0)(row, col) * error * this->d_activation(inputs.at(i)(row));
                  weights.at(0)(row, col) += rate * grad * inputs.at(i)(row);
               }
               biases.at(0)(col) += rate * grad;
            }
         }
      }
   }
   return &errors;
}

/**!Activation function
*/
double NeuralNetwork::activation(double input)
{
   double out = 0;
   if(input > 0){
      out = input;
   }
   return out;
}

/**!Vectorized activation function
*/
Eigen::VectorXd NeuralNetwork::activation(Eigen::VectorXd input)
{
   Eigen::VectorXd out = Eigen::VectorXd::Zero(input.size());
   #pragma omp parallel for
   for(auto i = 0; i < input.size(); ++i){
      out(i) = this->activation(input(i));
   }
   return out;
}

/**! Derivative of activation function
*/
double NeuralNetwork::d_activation(double input)
{
   double out = 0;
   if(input > 0){
      out = 1;
   }
   return out;
}

/**!Vectorized derivative of activation function
*/
Eigen::VectorXd NeuralNetwork::d_activation(Eigen::VectorXd input){
   Eigen::VectorXd out = Eigen::VectorXd::Zero(input.size());
   #pragma omp parallel for
   for(auto i = 0; i < input.size(); ++i){
      out(i) = this->d_activation(input(i));
   }
   return out;
}