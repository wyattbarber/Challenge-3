#ifndef _AUTOENCODER_HPP
#define _AUTOENCODER_HPP

#include "../Model.hpp"
#include "../basic/Activation.hpp"
#include "../optimizers/Optimizer.hpp"

using namespace optimization;

namespace neuralnet {

    template <typename T, ActivationFunc F, template<typename,typename> class C>
    class AutoEncoder : public Encoder<AutoEncoder<T, F, C>>
    {
    public:
        typedef T Scalar;
        typedef Eigen::Vector<T, Eigen::Dynamic> InputType;
        typedef InputType OutputType;
        typedef Eigen::Vector<T, Eigen::Dynamic> LatentType;

        AutoEncoder(int in_size, int latent_size) : w_update(in_size, latent_size), b_lt_update(latent_size), b_rc_update(in_size) { setup(in_size, latent_size); }
#ifndef NOPYTHON
        AutoEncoder(const py::tuple& data) : w_update(data[5]), b_lt_update(data[6]), b_rc_update(data[7])
        { 
            std::vector<T> w, bl, br;
            int in = data[0].cast<int>();
            int lt = data[1].cast<int>();

            setup(in, lt);
            w = data[2].cast<std::vector<T>>();
            bl = data[3].cast<std::vector<T>>();
            br = data[4].cast<std::vector<T>>();

            W = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(w.data(), in, lt);
            blt = Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(bl.data(), lt);
            brc = Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(br.data(), in);
        }
#endif

        template<typename X>
        OutputType forward(X&& input){ return decode(encode(input)); }

        template<typename X>
        InputType backward(X&& error){ return backward_encode(backward_decode(error)); }

        template<typename X>
        LatentType encode(X&& input);

        template<typename X>
        OutputType decode(X&& latent);

        template<typename X>
        LatentType backward_decode(X&& error);

        template<typename X>
        InputType backward_encode(X&& error);

        void update(double rate);

#ifndef NOPYTHON
        /** Pickling implementation
         *  
         * @return (in size, latent size, optimizer args..., weights, encoding biases, decoding biases)
         */
        py::tuple getstate() const;
#endif
    
    protected:
        size_t in_size, latent_size;

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> W;
        Eigen::Vector<T, Eigen::Dynamic> blt, brc;
        Eigen::Vector<T, Eigen::Dynamic> in, latent;
        Eigen::Vector<T, Eigen::Dynamic> zlt, zrc;
        Eigen::Vector<T, Eigen::Dynamic> alt, arc;
        Eigen::Vector<T, Eigen::Dynamic> drc, dlt;

        // Adam optimization data
        C<T,Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> w_update;
        C<T,LatentType> b_lt_update;
        C<T,OutputType> b_rc_update;

        void setup(int in_size, int latent_size)
        {
            this->in_size = in_size;
            this->latent_size = latent_size;
            
            // Apply he initialization
            this->W = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Random(in_size, latent_size).unaryExpr([in_size = in_size](T x)
                                                                                        { return x * std::sqrt(T(2) / static_cast<T>(in_size)); });
            this->blt = LatentType::Zero(latent_size);
            this->alt = LatentType::Zero(latent_size);
            this->dlt = LatentType::Zero(latent_size);
            this->zlt  = InputType::Zero(in_size);

            this->brc = InputType::Zero(in_size);
            this->arc = InputType::Zero(in_size);
            this->drc = InputType::Zero(in_size);
            this->zrc = LatentType::Zero(latent_size);

            this->in = InputType::Zero(in_size);
        }
    };
}


template <typename T, neuralnet::ActivationFunc F, template<typename,typename> class C>
template<typename X>
neuralnet::AutoEncoder<T, F, C>::LatentType neuralnet::AutoEncoder<T, F, C>::encode(X&& input)
{
    // Save input for this pass and calculate weighted signals
    in = {input};
    zlt = blt;
    zlt += W.transpose() * input;
    // Calculate and save activation function output
    alt =  Activation<Eigen::Dynamic, T, F>::f(zlt);
    return alt;
}


template <typename T, neuralnet::ActivationFunc F, template<typename,typename> class C>
template<typename X>
neuralnet::AutoEncoder<T, F, C>::OutputType neuralnet::AutoEncoder<T, F, C>::decode(X&& input)
{
    // Calculate and save activation function output
    zrc = brc;
    zrc += W * input;
    arc =  Activation<Eigen::Dynamic, T, F>::f(zrc);
    return arc;
}


template <typename T, neuralnet::ActivationFunc F, template<typename,typename> class C>
template<typename X>
neuralnet::AutoEncoder<T, F, C>::LatentType neuralnet::AutoEncoder<T, F, C>::backward_decode(X&& err)
{
    // Calculate this AutoEncoders error gradient
    drc = Activation<Eigen::Dynamic, T, F>::df(zrc, arc, err);
    // Calculate and return error gradient input to next AutoEncoder
    return W.transpose() * drc;
}


template <typename T, neuralnet::ActivationFunc F, template<typename,typename> class C>
template<typename X>
neuralnet::AutoEncoder<T, F, C>::InputType neuralnet::AutoEncoder<T, F, C>::backward_encode(X&& err)
{
    // Calculate this AutoEncoders error gradient
    dlt = Activation<Eigen::Dynamic, T, F>::df(zlt, alt, err);
    // Calculate and return error gradient input to next AutoEncoder
    return W * dlt;
}


template <typename T, neuralnet::ActivationFunc F, template<typename,typename> class C>
void neuralnet::AutoEncoder<T, F, C>::update(double rate)
{
    w_update.grad(rate, W, (in * dlt.transpose()) + (drc * alt.transpose()));
    b_lt_update.grad(rate, blt, dlt);
    b_rc_update.grad(rate, brc, drc);
}


#ifndef NOPYTHON
template <typename T, neuralnet::ActivationFunc F, template<typename,typename> class C>
py::tuple neuralnet::AutoEncoder<T, F, C>::getstate() const
{
    return py::make_tuple(
        W.rows(), W.cols(),
        std::vector<T>(W.data(), W.data() + W.size()),
        std::vector<T>(blt.data(), blt.data() + blt.size()),            
        std::vector<T>(brc.data(), brc.data() + brc.size()),
        w_update.getstate(),
        b_lt_update.getstate(),
        b_rc_update.getstate()
    );
}
#endif


#endif