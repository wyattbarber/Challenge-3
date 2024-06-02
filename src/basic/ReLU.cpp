#include "Layer.hpp"

template<>
Eigen::VectorXd neuralnet::Layer<neuralnet::ActivationFunc::ReLU>::forward(Eigen::VectorXd input)
{
    py::print(weights.col(0).format(Eigen::IOFormat(4, 0, ", ", "\n", "", "")));
    py::print("ReLU forward pass");
    set_z(input);
    py::print("Applying ReLU function");
    a = z;
    a.unaryExpr([](double x){ return (x > 0) ? x : 0.0; });

    // for (int i = 0; i < z.size(); ++i)
    // {
    //     py::print("ReLU node ", i);
    //     if (z(i) < 0.0)
    //     {
    //         a(i) = 0.0;
    //     }
    //     else
    //     {
    //         a(i) = z(i);
    //     }
    // }

    return a;
}

template<>
Eigen::VectorXd neuralnet::Layer<neuralnet::ActivationFunc::ReLU>::backward(Eigen::VectorXd err)
{
    // Calculate this layers error gradient
    d = Eigen::VectorXd::Zero(d.size());
    for (int i = 0; i < z.size(); ++i)
    {
        if (z(i) > 0.0)
        {
            d(i) = err(i);
        }
    }
    return weights * d;
}