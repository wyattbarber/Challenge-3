#ifndef _AUTOENCODER_HPP
#define _AUTOENCODER_HPP

#include "../Model.hpp"
#include "../basic/Activation.hpp"
#include "../optimizers/Optimizer.hpp"

using namespace optimization;

namespace neuralnet {

    template <typename T, ActivationFunc F, OptimizerClass C>
    class AutoEncoder : public Encoder<AutoEncoder<T, F, C>>
    {
    public:
        typedef Eigen::Vector<T, Eigen::Dynamic> InputType;
        typedef InputType OutputType;
        typedef Eigen::Vector<T, Eigen::Dynamic> LatentType;

        AutoEncoder(){ setup(0,0,0,0); }
        AutoEncoder(int in_size, int latent_size, double b1, double b2){ setup(in_size, latent_size, b1, b2); }
        AutoEncoder(int in_size, int latent_size)
        { 
            static_assert(C==OptimizerClass::None, "Adam parameters missing"); 
            setup(in_size, latent_size); 
        }
#ifndef NOPYTHON
        AutoEncoder(const py::tuple& data)
        { 
            std::vector<T> w, bl, br;
            int in = data[0].cast<int>();
            int lt = data[1].cast<int>();

            if constexpr (C == OptimizerClass::Adam)
            {
                setup(in, lt, data[2].cast<T>(), data[3].cast<T>()); 
                w = data[4].cast<std::vector<T>>();
                bl = data[5].cast<std::vector<T>>();
                br = data[6].cast<std::vector<T>>();
                adam::unpickle(data[7], adam_w);
                adam::unpickle(data[8], adam_blt);
                adam::unpickle(data[9], adam_brc);
            }
            else
            {
                setup(in, lt);
                w = data[2].cast<std::vector<T>>();
                bl = data[3].cast<std::vector<T>>();
                br = data[4].cast<std::vector<T>>();
            }

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
        adam::AdamData<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> adam_w;
        adam::AdamData<LatentType> adam_blt;
        adam::AdamData<OutputType> adam_brc;

        template<typename... Ts>
        void setup(Ts... Args)
        {
            auto args = std::tuple<Ts...>(Args...);

            this->in_size = std::get<0>(args);
            this->latent_size = std::get<1>(args);

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

            if constexpr (C == OptimizerClass::Adam)
            {
                adam_w.m = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(in_size, latent_size);
                adam_w.v = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(in_size, latent_size);
                adam_brc.m = InputType::Zero(in_size);
                adam_brc.v = InputType::Zero(in_size);                
                adam_blt.m = LatentType::Zero(latent_size);
                adam_blt.v = LatentType::Zero(latent_size);

                adam_w.b1 = std::get<2>(args);
                adam_w.b2 = std::get<3>(args);
                adam_brc.b1 = std::get<2>(args);
                adam_brc.b2 = std::get<3>(args);
                adam_blt.b1 = std::get<2>(args);
                adam_blt.b2 = std::get<3>(args);

                adam_w.b1powt = adam_w.b1;
                adam_w.b2powt = adam_w.b2;
                adam_brc.b1powt = adam_brc.b1;
                adam_brc.b2powt = adam_brc.b2;
                adam_blt.b1powt = adam_blt.b1;
                adam_blt.b2powt = adam_blt.b2;
            }
        }
    };
}


template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
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


template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
template<typename X>
neuralnet::AutoEncoder<T, F, C>::OutputType neuralnet::AutoEncoder<T, F, C>::decode(X&& input)
{
    // Calculate and save activation function output
    zrc = brc;
    zrc += W * input;
    arc =  Activation<Eigen::Dynamic, T, F>::f(zrc);
    return arc;
}


template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
template<typename X>
neuralnet::AutoEncoder<T, F, C>::LatentType neuralnet::AutoEncoder<T, F, C>::backward_decode(X&& err)
{
    // Calculate this AutoEncoders error gradient
    drc = Activation<Eigen::Dynamic, T, F>::df(zrc, arc, err);
    // Calculate and return error gradient input to next AutoEncoder
    return W.transpose() * drc;
}


template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
template<typename X>
neuralnet::AutoEncoder<T, F, C>::InputType neuralnet::AutoEncoder<T, F, C>::backward_encode(X&& err)
{
    // Calculate this AutoEncoders error gradient
    dlt = Activation<Eigen::Dynamic, T, F>::df(zlt, alt, err);
    // Calculate and return error gradient input to next AutoEncoder
    return W * dlt;
}


template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
void neuralnet::AutoEncoder<T, F, C>::update(double rate)
{
    if constexpr (C == OptimizerClass::Adam)
    {
        auto tmp = ((in * dlt.transpose()) + (drc * alt.transpose()));
        adam::adam_update_params(rate / 2.0, adam_w, W, tmp);
        adam::adam_update_params(rate, adam_blt, blt, dlt);
        adam::adam_update_params(rate, adam_brc, brc, drc);
    }
    else
    {
        W -= ((in * dlt.transpose()) + (drc * alt.transpose())) * (rate / 2.0);
        blt -= rate * dlt;
        brc -= rate * drc;
    }
}

#ifndef NOPYTHON
template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
py::tuple neuralnet::AutoEncoder<T, F, C>::getstate() const
{
    if constexpr (C == OptimizerClass::Adam)
    {
        return py::make_tuple(
            W.rows(), W.cols(),
            adam_w.b1, adam_w.b2,
            std::vector<T>(W.data(), W.data() + W.size()),
            std::vector<T>(blt.data(), blt.data() + blt.size()),            
            std::vector<T>(brc.data(), brc.data() + brc.size()),
            adam::pickle(adam_w),
            adam::pickle(adam_blt),
            adam::pickle(adam_brc)
        );
    }
    else
    {
        return py::make_tuple(
            W.rows(), W.cols(),
            std::vector<T>(W.data(), W.data() + W.size()),
            std::vector<T>(blt.data(), blt.data() + blt.size()),            
            std::vector<T>(brc.data(), brc.data() + brc.size())
        );
    }
}
#endif


#endif