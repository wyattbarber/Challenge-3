#include <pybind11/pybind11.h>
#include "neuralnet.hpp"
namespace py = pybind11;

NeuralNetwork::NeuralNetwork(std::vector<size_t> dims)
{
   this->n_layers = dims.size() - 1;

   for(size_t i = 1; i <= n_layers; ++i){
      Eigen::MatrixXd w_i = Eigen::MatrixXd::Random(dims.at(i-1), dims.at(i));
      w_i += Eigen::MatrixXd::Ones(dims.at(i-1), dims.at(i)); // Make initial weights non-negative, to avoid vanishing gradient
      this->weights.push_back(w_i);

      Eigen::VectorXd b_i = Eigen::VectorXd::Zero(dims.at(i));
      this->biases.push_back(b_i);

      Eigen::VectorXd h_i = Eigen::VectorXd::Zero(dims.at(i));
      this->z.push_back(h_i);
      this->a.push_back(h_i);
      this->d.push_back(h_i);
   }
}

Eigen::VectorXd NeuralNetwork::forwardPass(Eigen::VectorXd input)
{
#ifdef VERBOSE
   py::print("Calling forward pass...");
#endif

   z.at(0) = input * weights.at(0);
#ifdef VERBOSE
      py::print("Weighted inputs: ");
      for(int k = 0; k < z.at(0).size(); ++ k)
      {
         py::print('\t', z.at(0)(k));
      }
      py::print("");
#endif

   a.at(0) = this->activation(z.at(0));
#ifdef VERBOSE
      py::print("Input activation: ");
      for(int k = 0; k < a.at(0).size(); ++ k)
      {
         py::print('\t', a.at(0)(k));
      }
      py::print("");
#endif

   for(size_t i = 1; i < n_layers; ++i){
      z.at(i) = a.at(i-1) * weights.at(i);
      a.at(i) = this->activation(z.at(i));
#ifdef VERBOSE
      py::print("Hidden output ", i, ": ");
      for(int k = 0; k < a.at(i).size(); ++ k)
      {
         py::print('\t', a.at(i)(k));
      }
      py::print("");
#endif
   }
   return a.at(a.size()-1);
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
#ifdef VERBOSE
      py::print("Backpropagation pass number", iter);
#endif
      for(int i = 0; i < inputs.size(); ++i)
      {
         // Test forward pass and calculate error for this input set
#ifdef VERBOSE
         py::print("Calling forward pass and calculating error for input set ", i, " of ", inputs.size());
#endif
         Eigen::VectorXd output = this->forwardPass(inputs.at(i));
         error = output - outputs.at(i);
         errors.push_back(abs(d.at(n_layers-1)).sum());

#ifdef VERBOSE
         py::print("Propagating error");
#endif
         for(int j = n_layers-1; j >= 0 ; --j)
         {
            if(j == n_layers - 1)
            {
               d.at(j) = error;
            } else {
               Eigen::MatrixXd e = weights.at(j+1) * d.at(j+1);
               for(int k = 0; k < d.at(j).size(); ++k)
               {
                 d.at(j)(k) = e(k) * this->d_activation(z.at(j)(k)); 
               }
            }
         }

         // Update weights and biases
         for(int j = n_layers-1; j >= 0 ; --j)
         {
#ifdef VERBOSE
            py::print("Updating layer ", j+1, " of ", n_layers);
#endif
            // Iterate over all weights and nodes in this layer
            //#pragma omp parallel for
            for(int idx = 0; idx < weights.at(j).size(); ++idx)
            {
               int col = idx % weights.at(j).cols();
               int row = std::floor(idx / weights.at(j).cols());
#ifdef VERBOSE
               py::print("Updating weight ", row, " of node ", col, ". Starting value is ", weights.at(j)(row, col));
#endif
               // Get the input this weight modifies
               double a_in;
               if(j > 0)
               {
                  a_in = a.at(j-1)(row);
               } else {
                  a_in = inputs.at(i)(row);
               }

               weights.at(j)(row, col) -= rate * a_in * d.at(j)(col);
#ifdef VERBOSE
               py::print("Weight ", row, col, " updated to: ", weights.at(j)(row, col));
#endif

               if(row == 0)
               {
#ifdef VERBOSE
                  py::print("Updating bias for node ", col);
#endif
                  biases.at(j)(col) -= rate * d.at(j)(col);
               }
            }
         }
      }
   }
   return &errors;
}

/** Activation function
*/
double NeuralNetwork::activation(double input)
{
   double out = 0;
   if(input > 0){
      out = input;
   }
   return out;
}

/** Vectorized activation function
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

/** Derivative of activation function
*/
double NeuralNetwork::d_activation(double input)
{
   double out = 0;
   if(input > 0){
      out = 1;
   }
   return out;
}

/** Vectorized derivative of activation function
*/
Eigen::VectorXd NeuralNetwork::d_activation(Eigen::VectorXd input){
   Eigen::VectorXd out = Eigen::VectorXd::Zero(input.size());
   #pragma omp parallel for
   for(auto i = 0; i < input.size(); ++i){
      out(i) = this->d_activation(input(i));
   }
   return out;
}