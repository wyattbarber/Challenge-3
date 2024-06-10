#ifndef _ADAM_HPP
#define _ADAM_HPP

template <int I, int O, typename T>
void adam_update_params(double rate, double b1, double &b1powt, double b2, double &b2powt,
                        Eigen::Matrix<T, I, O> &m, Eigen::Matrix<T, I, O> &v, Eigen::Vector<T, O> &mb, Eigen::Vector<T, O> &vb,
                        Eigen::Matrix<T, I, O> &weights, Eigen::Vector<T, O> &biases, Eigen::Vector<T, I>& in, Eigen::Vector<T, O>& out, Eigen::Vector<T, O>& error_gradient)
{
    py::print("Adam update inner");
    const double epsilon = 1e-9;
    double decay1 = 1.0 - b1powt;
    double decay2 = 1.0 - b2powt;

    // Update weight moments
    py::print("Updating weight moments");
    Eigen::Matrix<T, I, O> weight_gradients = in * error_gradient.transpose();
    m = (b1 * m) + ((1.0 - b1) * weight_gradients);
    v = (b2 * v) + ((1.0 - b2) * weight_gradients.cwiseProduct(weight_gradients));
    Eigen::Matrix<T, I, O> mhat = m / decay1;
    Eigen::Matrix<T, I, O> vhat = (v / decay2).cwiseSqrt();
    py::print("Updating weights");
    weights -= rate * mhat.cwiseQuotient(vhat.unaryExpr([epsilon = epsilon](double x)
                                                        { return x + epsilon; }));

    // Update bias moments
    py::print("Updating bias moments");
    mb = (b1 * mb) + ((1.0 - b1) * error_gradient);
    vb = (b2 * vb) + ((1.0 - b2) * error_gradient.cwiseProduct(error_gradient));
    Eigen::Vector<T, O> mhat_b = mb / decay1;
    Eigen::Vector<T, O> vhat_b = (vb / decay2).cwiseSqrt();
    py::print("Updating biases");
    biases -= rate * mhat_b.cwiseQuotient(vhat_b.unaryExpr([epsilon = epsilon](double x)
                                                           { return x + epsilon; }));

    // Increment exponential decays
    b1powt *= b1;
    b2powt *= b2;
}

#endif