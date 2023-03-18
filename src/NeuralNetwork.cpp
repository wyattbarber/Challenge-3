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
      this->hidden.push_back(h_i);
   }
}

Eigen::VectorXd NeuralNetwork::forwardPass(Eigen::VectorXd input)
{
#ifdef VERBOSE
   py::print("Calling forward pass...");
#endif

   hidden.at(0) = input * weights.at(0);
   for(size_t i = 1; i < weights.size(); ++i){
      hidden.at(i) = this->activation(hidden.at(i-1)) * weights.at(i);
   }
   return this->activation(hidden.at(hidden.size()-1));
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
      for(size_t i = 0; i < inputs.size(); ++i)
      {
         // Test forward pass and calculate error for this input set
#ifdef VERBOSE
         py::print("Test point 1");
#endif
         Eigen::VectorXd output = this->forwardPass(inputs.at(i));
         double error = abs((output - outputs.at(i)).array()).sum();
         errors.push_back(error);

         // Iterate over layers
         for(size_t j = hidden.size()-1; j >= 1 ; --j)
         {
#ifdef VERBOSE
            py::print("Test point 2");
#endif
            // Iterate over each node in this layer
            for(size_t k = 0; k < hidden.at(j).size(); ++k)
            {
#ifdef VERBOSE
               py::print("Test point 3");
#endif
               Eigen::VectorXd grad = weights.at(j).row(k);
               Eigen::VectorXd a  = this->activation(hidden.at(j-1));
               Eigen::VectorXd d_a  = this->d_activation(hidden.at(j));
               for(size_t l = 0; l < grad.size(); ++l)
               {
#ifdef VERBOSE
                  py::print("Test point 4");
#endif
                  grad(l) *= error * d_a(l);
                  weights.at(j)(l,k) -= rate * a(l) * grad(l); 
                  biases.at(j)(l) -= rate * grad(l);
               }
            }
         }
         // Update input layer
         // Iterate over each node in this layer
         for(size_t k = 0; k < hidden.at(0).size(); ++k)
         {
#ifdef VERBOSE
            py::print("Test point 5");
#endif
            Eigen::VectorXd grad = weights.at(0).col(k);
            Eigen::VectorXd a  = this->activation(inputs.at(i));
            Eigen::VectorXd d_a  = this->d_activation(hidden.at(0));
            for(size_t l = 0; l < grad.size(); ++l)
            {
#ifdef VERBOSE
               py::print("Test point 6");
#endif
               grad(l) *= error * d_a(l);
               weights.at(0)(l,k) -= rate * a(l) * grad(l); 
               biases.at(0)(l) -= rate * grad(l);
            }
         }
      }      
   }
#ifdef VERBOSE
   py::print("Test point 7");
#endif
   return &errors;
}

Eigen::VectorXd NeuralNetwork::activation(Eigen::VectorXd input)
{
   for(auto i = input.begin(); i != input.end(); ++i){
      if(*i < 0){
         *i = 0;
      }
   }
   return input;
}

Eigen::VectorXd NeuralNetwork::d_activation(Eigen::VectorXd input){
   for(auto i = input.begin(); i != input.end(); ++i){
      if(*i < 0){
         *i = 0;
      } else {
         *i = 1;
      }
   }
   return input;
}