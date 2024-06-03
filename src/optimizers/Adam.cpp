#include "Adam.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;

optimization::Optimizer *optimization::Adam::copy()
{
    return new optimization::Adam(this->b1, this->b2);
}

void optimization::Adam::init(size_t in, size_t out)
{
    m = Eigen::MatrixXd::Zero(in, out);
    v = Eigen::MatrixXd::Zero(in, out);
    mb = Eigen::VectorXd::Zero(out);
    vb = Eigen::VectorXd::Zero(out);
}

void optimization::Adam::augment_gradients(Eigen::MatrixXd& weight_gradients, Eigen::VectorXd& bias_gradients)
{
    double decay1 = (1.0 - std::pow(b1, static_cast<double>(t)));
    double decay2 = (1.0 - std::pow(b2, static_cast<double>(t)));

    if(weight_gradients.rows() != m.rows())
        py::print("Gradient has wrong input size");
    if(weight_gradients.cols() != m.cols())
        py::print("Gradient has wrong output size");

    // Update weight moments
    m *= b1;
    m += (1.0 - b1) * weight_gradients;
    v *= b2;
    v += (1.0 - b2) * weight_gradients.cwiseProduct(weight_gradients);
    Eigen::MatrixXd mhat = m / decay1;
    Eigen::MatrixXd vhat = (v / decay2).cwiseSqrt();
    vhat.unaryExpr([](double x){return x == 0.0 ? 1e-9 : x;});
    weight_gradients = mhat.cwiseQuotient(vhat);

    // Update bias moments
    mb *= b1;
    mb += (1.0 - b1) * bias_gradients;
    vb *= b2;
    vb += (1.0 - b2) * bias_gradients.cwiseProduct(bias_gradients);
    Eigen::VectorXd mhat_b = mb / decay1;
    Eigen::VectorXd vhat_b = (vb / decay2).cwiseSqrt();
    vhat_b.unaryExpr([](double x){return x == 0.0 ? 1e-9 : x;});
    bias_gradients = mhat_b.cwiseQuotient(vhat_b);

    ++t;
}
