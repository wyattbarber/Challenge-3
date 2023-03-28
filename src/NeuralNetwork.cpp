#include <pybind11/pybind11.h>
#include "neuralnet.hpp"
namespace py = pybind11;
#include <cmath>
#include <assert.h>

#undef NDEBUG

NeuralNetwork::NeuralNetwork(std::vector<size_t> dims, std::vector<ActivationFunc> f)
{
   this->dims = dims;
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
//       py::print("Layer", i, "input -> output:");
//       for(int k = 0; k < a.at(i).size(); ++k)
//       {
//          py::print('\t', z.at(i)(k), "->", a.at(i)(k));
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

   for(int iter = 0; iter < passes; ++iter)
   {
// #ifdef VERBOSE
//       py::print("Backpropagation pass number", iter);
// #endif
      double e = 0.0;
      for(int i = 0; i < inputs.size(); ++i)
      {
         // Test forward pass and calculate error for this input set
         Eigen::VectorXd error = this->forwardPass(inputs.at(i)) - outputs.at(i);
         e += error.array().abs().sum() / inputs.size();
// #ifdef VERBOSE
//          py::print("Total error of backpropagation pass", iter, ", input", i, ":", abs(error.array()).sum());
// #endif

         this->err_propagate(error);
         this->param_propagate(rate);
      }
      errors.push_back(e);
   }
   return errors;
}


/** Vectorized activation function
*/
Eigen::VectorXd NeuralNetwork::activation(Eigen::VectorXd input, int layer)
{
   Eigen::VectorXd out = Eigen::VectorXd::Zero(input.size());
   switch(function_types.at(layer)){
      case ReLU:
         for(int i = 0; i < input.size(); ++i){
            if(input(i) > 0){
               out(i) = input(i);
            }
         }
         break;
      
      case Sigmoid:
         for(int i = 0; i < input.size(); ++i)
         {
            out(i) = 1 / (1 + std::exp(-1 * input(i)));
         }
         break;

      case SoftMax:
         // Caps unnormalized outputs at 1x10^300, to avoid getting errors due to exp() reporting infinity.
         out = input.array().exp().min(1e300).matrix();
         // py::print("Softmax");
         // for(int k = 0; k < out.size(); ++k)
         // {
         //    py::print('\t', out(k));
         // }
         // py::print("Sum:", out.array().min(1e300).sum());
         out /= out.array().min(1e300).sum();
         break;
     }
   return out;
}

/** Vectorized derivative of activation function
*/
Eigen::VectorXd NeuralNetwork::d_activation(Eigen::VectorXd input, int layer){
   Eigen::VectorXd out = Eigen::VectorXd::Zero(input.size());
   switch(function_types.at(layer)){
      case ReLU:
         for(int i = 0; i < input.size(); ++i){
            if(input(i) > 0){
               out(i) = 1;
            }
         }
         break;
      case Sigmoid:
         out = a.at(layer+1).cwiseProduct(Eigen::VectorXd::Ones(input.size()) - a.at(layer+1));
         break;
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

      // Calculate softmax derivative matrix 
      Eigen::MatrixXd d_sftm = Eigen::MatrixXd::Zero(z.at(j).size(), z.at(j).size());
      if(function_types.at(j-1) == ActivationFunc::SoftMax)
      {  
         for(int i = 0; i < z.at(j).size(); ++i)
         {
            for(int k = 0; k < z.at(j).size(); ++k)
            {
               if(i == k)
               {
                  d_sftm(i,k) = a.at(j)(i) * (1 - a.at(j)(k));
               } else {
                  d_sftm(i,k) = a.at(j)(i) * -a.at(j)(k);
               }
            }
         }
      }

      if(j == n_layers-1)
      {
         if(function_types.at(j-1) == ActivationFunc::SoftMax)
         {
            d.at(j) = d_sftm * error;
         } else {  
            d.at(j) = error.cwiseProduct(this->d_activation(z.at(j), j-1));
         }
      } else {
         if(function_types.at(j-1) == ActivationFunc::SoftMax)
         {
            d.at(j) = d_sftm * (weights.at(j) * d.at(j+1));
         } else {  
            d.at(j) = (weights.at(j) * d.at(j+1)).cwiseProduct(this->d_activation(z.at(j), j-1));
         }
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

void NeuralNetwork::reset()
{
   for(size_t i = 0; i < n_layers; ++i){
      if(i < n_layers-1)
      {
         weights.at(i) = Eigen::MatrixXd::Random(dims.at(i), dims.at(i+1));
      }

      biases.at(i) = Eigen::VectorXd::Zero(dims.at(i));
      z.at(i) = Eigen::VectorXd::Zero(dims.at(i));
      a.at(i) = Eigen::VectorXd::Zero(dims.at(i));
      d.at(i) = Eigen::VectorXd::Zero(dims.at(i));
   }
}