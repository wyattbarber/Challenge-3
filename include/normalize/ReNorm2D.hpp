#ifndef _RENORM2D_HPP
#define _RENORM2D_HPP

#include "../Model.hpp"
#include <vector>

namespace neuralnet
{
    /** Batch renormalization layer
     *
     *  Normalizes data to 0 mean and 1 standard deviation, by storing a
     *  moving average of the mean and deviation for the entire dataset.
     *  
     *  A learned linear transformation is also applied to the renormalized data.
     */
    template <typename T, OptimizerClass C>
    class ReNorm2D : public Model<ReNorm2D<T, C>>
    {
        public:
        typedef Eigen::Tensor<T, 3> InputType;
        typedef Eigen::Tensor<T, 3> OutputType;

        /** Constructs a normalization layer
         * 
         * @param N number of channels in the input tensor
         * @param rate update rate for the moving average of dataset distribution.
         * @param b1 Adam optimizer momentum decay rate.
         * @param b2 Adam optimizer velocity decay rate.
         */
        ReNorm2D(int N, T rate)
        {
            setup(N, rate);
        }        
        ReNorm2D(int N, T rate, T b1, T b2)
        {
            setup(N, rate, b1, b2);
        }
#ifndef NOPYTHON
        /** Unpickling constructor
         * 
         */
        ReNorm2D(py::tuple data)
        {
            if constexpr (C == OptimizerClass::Adam)
            {
                setup(data[0].cast<int>(), data[1].cast<T>(), data[1].cast<T>(), data[1].cast<T>());
                adam::unpickle(data[6], adam_lambda);
                adam::unpickle(data[7], adam_beta);
            }
            else
            {
                setup(data[0].cast<int>(), data[1].cast<T>());
            }
            mean =      Eigen::TensorMap<Eigen::Tensor<T,1>>(data[2].cast<std::vector<T>>().data(), N);
            dev =       Eigen::TensorMap<Eigen::Tensor<T,1>>(data[3].cast<std::vector<T>>().data(), N);
            lambda =    Eigen::TensorMap<Eigen::Tensor<T,1>>(data[4].cast<std::vector<T>>().data(), N);
            beta =         Eigen::TensorMap<Eigen::Tensor<T,1>>(data[5].cast<std::vector<T>>().data(), N);
        }
#endif

        template<typename X>      
        OutputType forward(X&& input);
        
        template<typename X>
        InputType backward(X&& error);

        void update(double rate);

#ifndef NOPYTHON
        /** Pickling implementation
         * 
         * Returns the following values in order:
         * * number of input channels
         * * moving average update rate
         * * recorded mean
         * * recorded standard deviation
         * * learned lambda (scale)
         * * learned beta (bias)
         * * (optimizer states...)
         *  
         * @return Model state
         */
        py::tuple getstate() const 
        { 
            if constexpr (C == OptimizerClass::Adam)
            {
                return py::make_tuple(N, avg_rate, 
                    std::vector<T>(mean.data(), mean.data() + mean.size()), 
                    std::vector<T>(dev.data(), mean.data() + mean.size()), 
                    std::vector<T>(lambda.data(), lambda.data() + lambda.size()), 
                    std::vector<T>(beta.data(), beta.data() + beta.size()),
                    adam::pickle(adam_lambda),
                    adam::pickle(adam_beta)
                    );
            }
            else
            {
                return py::make_tuple(N, avg_rate, 
                    std::vector<T>(mean.data(), mean.data() + mean.size()), 
                    std::vector<T>(dev.data(), mean.data() + mean.size()), 
                    std::vector<T>(lambda.data(), lambda.data() + lambda.size()), 
                    std::vector<T>(beta.data(), beta.data() + beta.size())
                    );
            }
        }
#endif
        protected:
        int N;
        T avg_rate;
        // Stored and learned parameters for each channel
        Eigen::Tensor<T,1> mean, dev, lambda, beta;
        // gradients
        Eigen::Tensor<T,1> grad_lambda, grad_beta;
        // Intermediates stored between forward and backward pass for gradient calculation
        static const inline T epsilon {1e-9};
        Eigen::Tensor<T,1> batch_dev, r;
        Eigen::Tensor<T,3> y, diff_sqr;
        // Optimizer data
        adam::AdamData<Eigen::Tensor<T,1>> adam_lambda, adam_beta;

        template<typename... Ts>
        void setup(Ts... Args)
        {
            auto args = std::tuple<Ts...>(Args...);
            N = std::get<0>(args);
            avg_rate = std::get<1>(args);
            grad_lambda = Eigen::Tensor<T,1>(N);
            grad_beta = Eigen::Tensor<T,1>(N);
            batch_dev - Eigen::Tensor<T,1>(N);
            r = Eigen::Tensor<T,1>(N);

            if constexpr (C == OptimizerClass::Adam)
            {
                adam_lambda.m = Eigen::Tensor<T,1>(N);
                adam_lambda.m.setZero();
                adam_lambda.v= Eigen::Tensor<T,1>(N);
                adam_lambda.v.setZero();
                adam_lambda.b1 = std::get<2>(args);
                adam_lambda.b2 = std::get<3>(args);
                adam_lambda.b1powt = adam_lambda.b1;
                adam_lambda.b2powt = adam_lambda.b2;

                adam_beta.m = Eigen::Tensor<T,1>(N);
                adam_beta.m.setZero();
                adam_beta.v= Eigen::Tensor<T,1>(N);
                adam_beta.v.setZero();
                adam_beta.b1 = std::get<2>(args);
                adam_beta.b2 = std::get<3>(args);
                adam_beta.b1powt = adam_beta.b1;
                adam_beta.b2powt = adam_beta.b2;
            }
        }

    };


    template <typename T, OptimizerClass C>
    template<typename X>      
    ReNorm2D<T,C>::OutputType ReNorm2D<T,C>::forward(X&& input)
    {
        auto M = input.dimension(0) * input.dimension(1);
        Eigen::Tensor<T,1> batch_mean(N);
        y = Eigen::Tensor<T,3>(input.dimensions());
        diff_sqr = Eigen::Tensor<T,3>(input.dimensions());
        Eigen::Tensor<T,0> res;

        for(int i = 0; i < N; ++i)
        {
            // Calculate
            res = input.chip(i,2).mean();
            batch_mean(i) = res(0);
            diff_sqr.chip(i,2) = (input.chip(i,2) - batch_mean(i)).square();
            res = diff_sqr.chip(i,2).sum();
            batch_dev(i) = std::sqrt((res(0) / M) + epsilon);

            // Renormalize input
            r(i) = batch_dev(i) / dev(i);
            auto d = (batch_mean(i) - mean(i)) / dev(i);
            auto xhat = (((input.chip(i,2) - batch_mean(i)) / batch_dev(i)) * r(i)) + d;

            // Transform to create output
            y.chip(i,2) = (lambda(i) * xhat) + beta(i);
        }

        // Update moving averages
        mean += avg_rate * (batch_mean - mean);
        dev += avg_rate * (batch_dev - dev);

        return y;
    }
    

    template <typename T, OptimizerClass C>
    template<typename X>
    ReNorm2D<T,C>::InputType ReNorm2D<T,C>::backward(X&& error)
    {
        auto M = error.dimension(0) * error.dimension(1);

        Eigen::Tensor<T,3> out;
     
        Eigen::Tensor<T,0> res;
        for(int i = 0; i < N; ++i)
        {
            // Calculate parameter gradients
            res = (y.chip(i,2) * error.chip(i,2)).sum();
            grad_lambda(i) = res(0);
            res = error.chip(i,2).sum();
            grad_beta(i) = res(0);
            // Backpropagate error
            auto a = diff_sqr.chip(i,2) * T(2) / batch_dev(i);
            auto b = a + (batch_dev(i) / M);
            out.chip(i,2) = b * -(r(i) / std::pow(batch_dev(i), T(2)));
        }

        return out;
    }


    template <typename T, OptimizerClass C>
    void ReNorm2D<T,C>::update(double rate)
    {
        if constexpr (C == OptimizerClass::Adam)
        {
            adam::adam_update_params(rate, adam_lambda, lambda, grad_lambda);
            adam::adam_update_params(rate, adam_beta, beta, grad_beta);
        }
        else
        {
            lambda -= rate * grad_lambda;
            beta -= rate * grad_beta;
        }
    }
}
#endif