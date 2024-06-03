#ifndef _ADAM_HPP
#define _ADAM_HPP

#include "../Optimizer.hpp"

namespace optimization
{
    class Adam : public Optimizer
    {
    public:
        Adam(double b1, double b2)
        {
            this->b1 = b1;
            this->b2 = b2;
            this->t = 1;
        }

        Optimizer *copy();

        void init(size_t in, size_t out);

        /** Applies optimization algorithm in-place to gradients
         *
         *
         */
        void augment_gradients(Eigen::MatrixXd& weight_gradients, Eigen::VectorXd& bias_gradients);


    protected:
        double b1, b2;
        unsigned t;
        Eigen::MatrixXd m, v;
        Eigen::VectorXd mb, vb;
    };
}

#endif