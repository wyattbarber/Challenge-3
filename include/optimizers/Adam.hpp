#ifndef _ADAM_HPP
#define _ADAM_HPP

#include "Optimizer.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#ifndef NOPYTHON
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif


namespace optimization
{
    template <typename P, P::Scalar B1, P::Scalar B2, template<typename> class Next = NoOpt, typename enabler = bool>
    class Adam : public Optimizer<Adam<P,B1,B2,Next>>{};

    template <typename P, P::Scalar B1, P::Scalar B2, template<typename> class Next>
    class Adam<P, B1, B2, Next, std::enable_if_t<std::is_base_of_v<Eigen::MatrixBase<P>,P>, bool>>: public Optimizer<Adam<P,B1,B2,Next,bool>>
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            typedef P::Scalar Scalar;   

            template<typename... Ts>
            Adam(Eigen::Index dim0, Ts... Dims) : 
                m(dim0, Dims...), 
                v(dim0, Dims...), 
                next(dim0, Dims...)
            {
                b1powt = B1;
                b2powt = B2;
            }
            Adam(const py::tuple& data) : next(data[5])
            {
                std::cout << "Unpickling adam data" << std::endl;

                auto dims = data[1].cast<Eigen::array<Eigen::Index,2>>();
                auto _m = data[2].cast<std::vector<Scalar>>();
                auto _v = data[3].cast<std::vector<Scalar>>();
                b1powt = std::pow(B1, data[4].cast<unsigned>());
                b2powt = std::pow(B2, data[4].cast<unsigned>());

                std::cout << "Unpickling adam state matrices" << std::endl;
                
                m = Eigen::Map<P>(m.data(), dims[0], dims[1]);
                v = Eigen::Map<P>(v.data(), dims[0], dims[1]);
            }
            
            template<typename X>
            void grad(Scalar rate, P& params, X&& gradient)
            {
                Scalar decay1 = Scalar(1) - b1powt;
                Scalar decay2 = Scalar(1) - b2powt;

                // Update weight moments
                m = (B1 * m) + ((Scalar(1) - B1) * gradient);
                v = (B2 * v) + ((Scalar(1) - B2) * gradient.cwiseProduct(gradient));
                auto mhat = m / decay1;
                auto vhat = (v / decay2).cwiseSqrt();

                // Apply nested updates
                next.grad(rate, params, 
                    mhat.cwiseQuotient(vhat.unaryExpr([](Scalar x)
                        { return x + Eigen::NumTraits<Scalar>::epsilon(); })
                        )
                    );
                // Increment exponential decays
                b1powt *= B1;
                b2powt *= B2;
            }

#ifndef NOPYTHON
            /** Encodes constructor and state info
             * 
             * * Parameter rank
             * * Dimension vector
             * * Moment values
             * * Velocity values
             * * Timestep
             * * Nested optimizer state
             * 
             * @return tuple containing the above data in order
             */
            py::tuple getstate() const 
            { 
                return py::make_tuple(
                    2,
                    Eigen::array<Eigen::Index,2>{m.rows(), m.cols()},
                    std::vector<Scalar>(m.data(), m.data() + m.size()),
                    std::vector<Scalar>(v.data(), v.data() + v.size()),
                    static_cast<unsigned>(std::log(b1powt) / std::log(B1)),
                    next.getstate()
                );
            }
#endif
        protected:
            P m, v;
            Scalar b1powt, b2powt;
            Next<P> next;
    };

    /** Specialization for tensor types */
    template <typename P, P::Scalar B1, P::Scalar B2, template<typename> class Next>
    class Adam<P, B1, B2, Next, std::enable_if_t<std::is_base_of_v<Eigen::TensorBase<P>,P>, bool>> : public Optimizer<Adam<P,B1,B2,Next,bool>>
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            typedef P::Scalar Scalar;   

            template<typename... Ts>
            Adam(Eigen::Index dim0, Ts... Dims) : 
                m(dim0, Dims...), 
                v(dim0, Dims...), 
                next(dim0, Dims...)
            {
                b1powt = B1;
                b2powt = B2;
            }
            Adam(const py::tuple& data) : next(data[5])
            {
                auto dims = data[1].cast<Eigen::array<Eigen::Index,P::NumDimensions>>();
                auto _m = data[2].cast<std::vector<Scalar>>();
                auto _v = data[3].cast<std::vector<Scalar>>();
                b1powt = std::pow(B1, data[4].cast<unsigned>());
                b2powt = std::pow(B2, data[4].cast<unsigned>());
                
                m = Eigen::TensorMap<P>(m.data(), dims);
                v = Eigen::TensorMap<P>(v.data(), dims);
            }
            
            template<typename X>
            void grad(Scalar rate, P& params, X&& gradient)
            {
                Scalar decay1 = Scalar(1) - b1powt;
                Scalar decay2 = Scalar(1) - b2powt;

                // Update weight moments
                m = (B1 * m) + ((Scalar(1) - B1) * gradient);
                v = (B2 * v) + ((Scalar(1) - B2) * gradient.square());
                auto mhat = m / decay1;
                auto vhat = (v / decay2).sqrt();
                
                // Apply nested updates
                next.grad(rate, params, 
                    mhat / (vhat + Eigen::NumTraits<Scalar>::epsilon())
                    );

                // Increment exponential decays
                b1powt *= B1;
                b2powt *= B2;
            }

#ifndef NOPYTHON
            py::tuple getstate() const 
            { 
                return py::make_tuple(
                    P::NumDimensions,
                    m.dimensions(),
                    std::vector<Scalar>(m.data(), m.data() + m.size()),
                    std::vector<Scalar>(v.data(), v.data() + v.size()),
                    static_cast<unsigned>(std::log(b1powt) / std::log(B1)),
                    next.getstate()
                );
            }
#endif
        protected:
            P m, v;
            Scalar b1powt, b2powt;
            Next<P> next;
    };
}
 
#endif