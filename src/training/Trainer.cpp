#include "Trainer.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;


template<int I, int O, typename T>
std::vector<double> training::Trainer<I, O, T>::train(unsigned N, double rate)
{
    std::vector<double> avg_err;

    // Total output size for normalizing errors
    double out_norm = 0.0;
    for (int i = 0; i < outputs.size(); ++i)
    {
        out_norm += outputs[i].norm();
    }

    
    for (int iter = 0; iter < N; ++iter)
    {
        double e = 0.0;
        for (int i = 0; i < inputs.size(); ++i)
        {
            // Test forward pass and calculate error for this input set
            Eigen::Vector<T, I> in = inputs[i];
            Eigen::Vector<T, O> out = *(model.forward(in));
            Eigen::Vector<T, O> error = out - outputs[i];
            e += error.norm();
            // Run backward pass
            model.backward(error);
            // Update model
            model.update(rate);
        }
        py::print("Epoch", iter, "average loss:", e / out_norm);
        avg_err.push_back(e / out_norm);
    }
    return avg_err;
}