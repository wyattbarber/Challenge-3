#ifndef _ADAM_HPP
#define _ADAM_HPP

#include "../Optimizer.hpp"

namespace optimization
{
    class Adam : public Optimizer
    {
    public:
        Adam(double b1, double b2, double epsilon = 1e-9)
        {
            this->b1 = b1;
            this->b2 = b2;
            this->b1powt = b1;
            this->b2powt = b2;
            this->minusb1 = 1.0 - b1;
            this->minusb2 = 1.0 - b2;
            this->epsilon = epsilon;
        }

        Optimizer *copy();

        void init(size_t in, size_t out);

        void augment_gradients(Eigen::MatrixXd& weight_gradients, Eigen::VectorXd& bias_gradients);

        void reset();

    protected:
        double b1, b2, minusb1, minusb2, b1powt, b2powt;
        double epsilon;
        Eigen::MatrixXd m, v;
        Eigen::VectorXd mb, vb;
    };
}

#endif