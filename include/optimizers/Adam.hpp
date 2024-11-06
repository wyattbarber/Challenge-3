#ifndef _ADAM_HPP
#define _ADAM_HPP

namespace optimization
{
    namespace adam
    {
        template <typename MatType>
        class AdamData
        {
            public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            AdamData(){}
            ~AdamData(){}

            double b1;
            double b2;
            double b1powt;
            double b2powt;
            double epsilon = 1e-9;
            MatType m;
            MatType v;
        };

        template <typename Derived, typename DerivedGrad>
        void adam_update_params(double rate, AdamData<Derived> &data, Eigen::MatrixBase<Derived> &params, Eigen::MatrixBase<DerivedGrad> &gradient)
        {
            double decay1 = 1.0 - data.b1powt;
            double decay2 = 1.0 - data.b2powt;

            // Update weight moments
            data.m = (data.b1 * data.m) + ((1.0 - data.b1) * gradient);
            data.v = (data.b2 * data.v) + ((1.0 - data.b2) * gradient.cwiseProduct(gradient));
            auto mhat = data.m / decay1;
            auto vhat = (data.v / decay2).cwiseSqrt();
            params -= rate * mhat.cwiseQuotient(vhat.unaryExpr([epsilon = data.epsilon](double x)
                                                               { return x + epsilon; }));
            // Increment exponential decays
            data.b1powt *= data.b1;
            data.b2powt *= data.b2;
        }
    }
}
#endif