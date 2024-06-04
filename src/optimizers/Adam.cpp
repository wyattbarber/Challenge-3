#include "Adam.hpp"
#include <cstdlib>
#include <cmath>
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


void optimization::Adam::reset()
{
    b1powt = b1;
    b2powt = b2;            
    m = Eigen::MatrixXd::Zero(m.rows(), m.cols());
    v = Eigen::MatrixXd::Zero(v.rows(), v.cols());
    mb = Eigen::VectorXd::Zero(mb.size());
    vb = Eigen::VectorXd::Zero(vb.size());
}


static const double epsilon = 1e-9; /// Smallest value to allow in denominators, for stability
void optimization::Adam::augment_gradients(Eigen::MatrixXd& weight_gradients, Eigen::VectorXd& bias_gradients)
{
    double decay1 = 1.0 - b1powt;
    double decay2 = 1.0 - b2powt;

    // Update weight moments
    m *= b1;
    m += minusb1 * weight_gradients;
    v *= b2;
    v += minusb2 * weight_gradients.cwiseAbs2();
    Eigen::MatrixXd mhat = m / decay1;
    Eigen::MatrixXd vhat = (v / decay2).cwiseSqrt();
    vhat.unaryExpr([](double x){return 1.0 / (abs(x) < epsilon ? (epsilon * (std::signbit(x) ? -1.0 : 1.0)) : x);});
    weight_gradients = mhat.cwiseProduct(vhat);

    // Update bias moments
    mb *= b1;
    mb += minusb1 * bias_gradients;
    vb *= b2;
    vb += minusb2 * bias_gradients.cwiseAbs2();
    Eigen::VectorXd mhat_b = mb / decay1;
    Eigen::VectorXd vhat_b = (vb / decay2).cwiseSqrt();
    vhat.unaryExpr([](double x){return 1.0 / (abs(x) < epsilon ? (epsilon * (std::signbit(x) ? -1.0 : 1.0)) : x);});
    bias_gradients = mhat_b.cwiseProduct(vhat_b);

    // Increment exponential decays
    b1powt *= b1;
    b2powt *= b2;
}
