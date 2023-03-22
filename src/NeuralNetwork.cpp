#include <pybind11/pybind11.h>
#include "neuralnet.hpp"
namespace py = pybind11;
#include <cmath>

NeuralNetwork::NeuralNetwork(std::vector<size_t> dims, std::vector<ActivationFunc> f)
{
   this->n_layers = dims.size();
   this->function_types = f;

   for(size_t i = 0; i < n_layers; ++i){
      if(i < n_layers-1)
      {
         this->weights.push_back(Eigen::MatrixXd::Random(dims.at(i), dims.at(i+1)));
      }

      this->biases.push_back(Eigen::VectorXd::Zero(dims.at(i)));
      this->z.push_back(Eigen::VectorXd::Zero(dims.at(i)));
      this->a.push_back(Eigen::VectorXd::Zero(dims.at(i)));
      this->d.push_back(Eigen::VectorXd::Zero(dims.at(i)));
   }

// #ifdef VERBOSE
//    py::print("Constructed neural network with", n_layers, "layers");
//    for(int i = 0; i < z.size(); ++i)
//    {
//       py::print('\t', "Layer", i, "size", z.at(i).size());
//    }
//    py::print();
// #endif
}

Eigen::VectorXd NeuralNetwork::forwardPass(Eigen::VectorXd input)
{
// #ifdef VERBOSE
//    py::print("Calling forward pass...");
// #endif

   for(size_t i = 0; i < n_layers; ++i){
      if(i == 0)
      {
         a.at(0) = input;
      } else {
         z.at(i) = a.at(i-1).transpose() * weights.at(i-1);
         a.at(i) = this->activation(z.at(i) + biases.at(i), i-1); 
      }
// #ifdef VERBOSE
//       py::print("Layer", i, "output : ");
//       for(int k = 0; k < a.at(i).size(); ++k)
//       {
//          py::print('\t', a.at(i)(k));
//       }
//       py::print("");
// #endif
   }
   return a.at(n_layers-1);
}

std::vector<double> NeuralNetwork::train(std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> outputs, double rate, int passes)
{
// #ifdef VERBOSE
//    py::print("Calling back propagation...");
// #endif

   std::vector<double> errors;
   errors.reserve(passes * inputs.size());

   for(int iter = 0; iter < passes; ++iter)
   {
// #ifdef VERBOSE
//       py::print("Backpropagation pass number", iter);
// #endif
      for(int i = 0; i < inputs.size(); ++i)
      {
         // Test forward pass and calculate error for this input set
         Eigen::VectorXd error = this->forwardPass(inputs.at(i)) - outputs.at(i);
         errors.push_back(abs(error.array()).sum());
// #ifdef VERBOSE
//          py::print("Total error of backpropagation pass", iter, ", input", i, ":", abs(error.array()).sum());
// #endif

         this->err_propagate(error);
         this->param_propagate(rate);
      }

   }
   return errors;
}

/** Activation function
*/
double NeuralNetwork::activation(double input, int layer)
{
   double out = 0;
   switch(function_types.at(layer)){
      case ReLU:
         if(input > 0){
            out = input;
         }
         break;
      case Sigmoid:
         out = 1 / (1 + std::exp(-1 * input));
         break;
   }
   return out;
}

/** Vectorized activation function
*/
Eigen::VectorXd NeuralNetwork::activation(Eigen::VectorXd input, int layer)
{
   Eigen::VectorXd out = Eigen::VectorXd::Zero(input.size());
   //#pragma omp parallel for
   for(auto i = 0; i < input.size(); ++i){
      out(i) = this->activation(input(i), layer);
   }
   return out;
}

/** Derivative of activation function
*/
double NeuralNetwork::d_activation(double input, int layer)
{
   double out = 0;
   switch(function_types.at(layer)){
      case ReLU:
         if(input > 0){
            out = 1;
         }
         break;
      case Sigmoid:
         double a = this->activation(input, layer);
         out = a * (1 - a);
         break;
   }
   return out;
}

/** Vectorized derivative of activation function
*/
Eigen::VectorXd NeuralNetwork::d_activation(Eigen::VectorXd input, int layer){
   Eigen::VectorXd out = Eigen::VectorXd::Zero(input.size());
   //#pragma omp parallel for
   for(auto i = 0; i < input.size(); ++i){
      out(i) = this->d_activation(input(i), layer);
   }
   return out;
}


void NeuralNetwork::err_propagate(Eigen::VectorXd error)
{
   for(int j = n_layers-1; j > 0 ; --j)
   {
// #ifdef VERBOSE
//       py::print("Propagating error over layer", j+1);
// #endif
      if(j == n_layers-1)
      {
         d.at(j) = error.cwiseProduct(this->d_activation(z.at(j), j-1));
      } else {
         d.at(j) = (weights.at(j) * d.at(j+1)).cwiseProduct(this->d_activation(z.at(j), j-1));
      }
// #ifdef VERBOSE
//       py::print("Sum of error at layer", j+1, ":", abs(d.at(j).array()).sum());
// #endif
   }
}

void NeuralNetwork::param_propagate(double rate)
{
   // Iterate over layers
   for(int l = 0; l < n_layers-1; ++l)
   {
      // Iterate over nodes in the layer these weights input to
      for(int n = 0; n < a.at(l+1).size(); ++ n)
      {
// #ifdef VERBOSE
//          py::print("Updating parameters of node", n, "in layer", l, "by", rate * d.at(l+1)(n));
// #endif
         // Update weights
         weights.at(l).col(n) -= rate * d.at(l+1)(n) * a.at(l);

         // Update bias
         biases.at(l+1)(n) -= rate * d.at(l+1)(n);
      }
   }  
}