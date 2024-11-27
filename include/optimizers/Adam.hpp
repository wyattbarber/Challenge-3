#ifndef _ADAM_HPP
#define _ADAM_HPP

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#ifndef NOPYTHON
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif


namespace optimization
{
    namespace adam
    {
        template <typename MatType>
        class AdamData
        {
            public:
            typedef MatType::Scalar Scalar;

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            AdamData(){}
            ~AdamData(){}

            Scalar b1;
            Scalar b2;
            Scalar b1powt;
            Scalar b2powt;
            Scalar epsilon = Eigen::NumTraits<Scalar>::epsilon();
            MatType m;
            MatType v;
        };

#ifndef NOPYTHON
        /** Serializes an Adam state struct.
         * 
         * Stores momentum, velocity, and timestep info in a way that can be pickled
         * with pybind11. The weights b1 and b2 are not included, as they are expected to be 
         * part of the data already pickled with a models constructor args.
         * 
         * The data is stored in the returned tuple in the following order:
         * 1. Momentum
         * 2. Velocity
         * 3. timestep
         * 
         * @param data Adam state to serialize
         * 
         * @return py::tuple containing serializable state data 
         */
        template<typename MatType>
        py::tuple pickle(const AdamData<MatType>& data)
        {
            using T = MatType::Scalar;
            return py::make_tuple(
                std::vector<T>(data.m.data(), data.m.data() + data.m.size()),
                std::vector<T>(data.v.data(), data.v.data() + data.v.size()),
                static_cast<unsigned>(std::log(data.b1powt) / std::log(data.b1))
            );
        }

        /** Restores Adam state from serialized data.
         * 
         * @param pickled Output from a call to adam::pickle
         * @param data Adam state struct to restore with pickled data
         */
        template<typename MatType>
        void unpickle(const py::tuple& pickled, AdamData<MatType>& data)
        {
            using T = MatType::Scalar;
            std::vector<T> m = pickled[0].cast<std::vector<T>>();
            std::vector<T> v = pickled[1].cast<std::vector<T>>();
            unsigned t = pickled[2].cast<unsigned>();

            data.b1powt = std::pow(data.b1, t);
            data.b2powt = std::pow(data.b2, t);

            if constexpr (std::is_base_of_v<Eigen::TensorBase<MatType>, MatType>)
            {
                data.m = Eigen::TensorMap<MatType>(m.data(), data.m.dimensions());
                data.v = Eigen::TensorMap<MatType>(v.data(), data.v.dimensions());
            }
            else
            {
                data.m = Eigen::Map<MatType>(m.data(), data.m.rows(), data.m.cols());
                data.v = Eigen::Map<MatType>(v.data(), data.v.rows(), data.v.cols());
            }
        }

#endif

        template <typename Derived, typename DerivedGrad>
        void adam_update_params(double rate, AdamData<Derived> &data, Eigen::MatrixBase<Derived> &params, Eigen::MatrixBase<DerivedGrad>& gradient)
        {
            using S = AdamData<Derived>::Scalar;

            S decay1 = 1.0 - data.b1powt;
            S decay2 = 1.0 - data.b2powt;

            // Update weight moments
            data.m = (data.b1 * data.m) + ((S(1) - data.b1) * gradient);
            data.v = (data.b2 * data.v) + ((S(1) - data.b2) * gradient.cwiseProduct(gradient));
            auto mhat = data.m / decay1;
            auto vhat = (data.v / decay2).cwiseSqrt();
            params -= rate * mhat.cwiseQuotient(vhat.unaryExpr([](S x)
                                                               { return x + Eigen::NumTraits<S>::epsilon(); }));
            // Increment exponential decays
            data.b1powt *= data.b1;
            data.b2powt *= data.b2;
        }

        template <typename Derived, typename DerivedGrad>
        void adam_update_params(double rate, AdamData<Derived> &data, Eigen::TensorBase<Derived> &params, Eigen::TensorBase<DerivedGrad>& gradient)
        {
            using S = AdamData<Derived>::Scalar;

            S decay1 = 1.0 - data.b1powt;
            S decay2 = 1.0 - data.b2powt;

            // Update weight moments
            data.m = (data.b1 * data.m) + ((S(1) - data.b1) * gradient);
            data.v = (data.b2 * data.v) + ((S(1) - data.b2) * gradient.square());
            auto mhat = data.m / decay1;
            auto vhat = (data.v / decay2).sqrt();
            params -= rate * mhat / (vhat + Eigen::NumTraits<S>::epsilon());
            // Increment exponential decays
            data.b1powt *= data.b1;
            data.b2powt *= data.b2;
        }
    }
}
 
#endif