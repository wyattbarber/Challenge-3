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
    template <typename T, template<typename> class C>
    class ReNorm2D : public Model<ReNorm2D<T, C>>
    {
        public:
        typedef T Scalar;
        typedef Eigen::Tensor<T, 3> InputType;
        typedef Eigen::Tensor<T, 3> OutputType;

        /** Constructs a normalization layer
         * 
         * @param N number of channels in the input tensor
         * @param rate update rate for the moving average of dataset distribution.
         * @param b1 Adam optimizer momentum decay rate.
         * @param b2 Adam optimizer velocity decay rate.
         */
        ReNorm2D(int N, T rate) :   
            mean(N,1),
            dev(N,1),
            lambda(N,1),
            beta(N,1),
            grad_lambda(N,1),
            grad_beta(N,1),
            batch_dev(N,1),
            r(N,1),
            update_lambda(N,1),
            update_beta(N,1)
        {
            setup(N, rate);
        }
        

#ifndef NOPYTHON
        /** Unpickling constructor
         * 
         */
        ReNorm2D(const py::tuple& data) : 
            N(data[0].cast<int>()),
            mean(N,1),
            dev(N,1),
            lambda(N,1),
            beta(N,1),
            grad_lambda(N,1),
            grad_beta(N,1),
            batch_dev(N,1),
            r(N,1),
            update_lambda(data[6]),
            update_beta(data[7])
        {
            setup(data[0].cast<int>(), data[1].cast<T>());

            // Trying to do these unpacking operations in one line each seems to cause segfaults
            auto m = data[2].cast<std::vector<T>>();
            mean = Eigen::TensorMap<Eigen::Tensor<T,2>>(m.data(), N, 1);
            
            auto d = data[3].cast<std::vector<T>>();
            dev = Eigen::TensorMap<Eigen::Tensor<T,2>>(d.data(), N, 1);

            auto l = data[4].cast<std::vector<T>>();
            lambda = Eigen::TensorMap<Eigen::Tensor<T,2>>(l.data(), N, 1);

            auto b = data[5].cast<std::vector<T>>();
            beta = Eigen::TensorMap<Eigen::Tensor<T,2>>(b.data(), N, 1);
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
         * * (optimizer constructor args...)
         *  
         * @return Model state
         */
        py::tuple getstate() const 
        { 
            return py::make_tuple(N, avg_rate,
                std::vector<T>(mean.data(), mean.data() + mean.size()), 
                std::vector<T>(dev.data(), dev.data() + dev.size()), 
                std::vector<T>(lambda.data(), lambda.data() + lambda.size()), 
                std::vector<T>(beta.data(), beta.data() + beta.size()),
                update_lambda.getstate(),
                update_beta.getstate()
                );
        }
#endif
        protected:
        int N;
        T avg_rate;
        /** @note 
            Rank two tensors are used for 1D data due to issues using 1D tensor. Using () operator 
            in operations results in assertions failing due to index out of bounds, but using it
            to simply access data such as `cout << tensor(0)` succeeds. Haven't figured out what was causing
            or tried reverting it after fixing other issues.
        */
        // Stored and learned parameters for each channel
        Eigen::Tensor<T,2> mean, dev, lambda, beta;
        // gradients
        Eigen::Tensor<T,2> grad_lambda, grad_beta;
        // Intermediates stored between forward and backward pass for gradient calculation
        Eigen::Tensor<T,2> batch_dev, r;
        Eigen::Tensor<T,3> xhat, diff;
        // Optimizer data
        C<Eigen::Tensor<T,2>> update_lambda, update_beta;
        // Timestep and relaxation rate for limits on r and d


        void setup(int N, T avg_rate)
        {
            this->N = N;
            this->avg_rate = avg_rate;
           
            mean.setZero();
            dev.setConstant(T(1));
            lambda.setConstant(T(1));
            beta.setZero();
        }
    };


    template <typename T, template<typename> class C>
    template<typename X>      
    ReNorm2D<T,C>::OutputType ReNorm2D<T,C>::forward(X&& input)
    {
        auto M = static_cast<T>(input.dimension(0) * input.dimension(1));
        auto y = Eigen::Tensor<T,3>(input.dimensions());
        xhat = Eigen::Tensor<T,3>(input.dimensions());
        diff = Eigen::Tensor<T,3>(input.dimensions());
        Eigen::Tensor<T,0> res;
        Eigen::Tensor<T,2> batch_mean(N,1);

#ifndef NDEBUG
        std::cout << "Renormalizing "<<
            input.dimension(0)<<'x'<<input.dimension(1)<<'x'<<input.dimension(2)<<" tensor" << std::endl;
#endif

        for(int i = 0; i < N; ++i)
        {
            // Calculate stats
            res = input.chip(i,2).mean();
            batch_mean(i,0) = res(0);
            diff.chip(i,2) = (input.chip(i,2) - batch_mean(i,0));
            res = diff.chip(i,2).square().sum();
            using std::sqrt;
            batch_dev(i,0) = sqrt((res(0) / M)  + Eigen::NumTraits<T>::epsilon());

            // Renormalize input
            r(i,0) = batch_dev(i,0) / (dev(i,0) + Eigen::NumTraits<T>::epsilon());
            auto d = (batch_mean(i,0) - mean(i,0)) / (dev(i,0) + Eigen::NumTraits<T>::epsilon());
            xhat.chip(i,2) = (((diff.chip(i,2) / batch_dev(i,0)) * r(i,0)) + d)
                .cwiseMin(Eigen::NumTraits<T>::highest()).cwiseMax(Eigen::NumTraits<T>::lowest());

            // Transform to create output
            y.chip(i,2) = (lambda(i,0) * xhat.chip(i,2)) + beta(i,0);
        }

        // Update moving averages
        if(this->train_mode)
        {
            mean += avg_rate * (batch_mean - mean);
            dev += avg_rate * (batch_dev - dev);
        }
        
        return y;
    }
    

    template <typename T, template<typename> class C>
    template<typename X>
    ReNorm2D<T,C>::InputType ReNorm2D<T,C>::backward(X&& error)
    {
#ifndef NDEBUG
        std::cout << "ReNorm2D backpropagating "<<
            error.dimension(0)<<'x'<<error.dimension(1)<<'x'<<error.dimension(2)<<" gradient" << std::endl;
#endif
        auto M = static_cast<T>(error.dimension(0) * error.dimension(1));

        Eigen::Tensor<T,3> out(error.dimensions());
        Eigen::Tensor<T,2> batch_dev_sqr = batch_dev.square();
        Eigen::Tensor<T,0> res;

        for(int i = 0; i < N; ++i)
        {
            // Calculate parameter gradients
            res = (xhat.chip(i,2) * error.chip(i,2)).sum();
            grad_lambda(i,0) = res(0);
            res = error.chip(i,2).sum();
            grad_beta(i,0) = res(0);
            // Backpropagate error
            auto dl_dxh = error.chip(i,2) * lambda(i,0);
            auto dl_ds_1 = (diff.chip(i,2) * dl_dxh).sum();
            Eigen::Tensor<T,0> dl_ds = -(r(i,0) / batch_dev_sqr(i,0)) * dl_ds_1;
            auto dl_dm = -(r(i,0) / batch_dev(i,0)) * dl_dxh.sum();
            Eigen::Tensor<T,2> a = dl_dxh * r(i,0) / batch_dev(i,0);
            T _b = dl_ds(0) / (M * batch_dev(i,0));
            Eigen::Tensor<T,2> b = _b * diff.chip(i,2);
            Eigen::Tensor<T,0> c = dl_dm / M;
            out.chip(i,2) = (a + b) + c(0);
        }
        return out;
    }


    template <typename T, template<typename> class C>
    void ReNorm2D<T,C>::update(double rate)
    {
        update_lambda.grad(rate, lambda, grad_lambda);
        update_beta.grad(rate, beta, grad_beta);
    }
}
#endif