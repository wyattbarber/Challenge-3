#ifndef _LAYER_HPP
#define _LAYER_HPP

#include "../Model.hpp"
#include "../optimizers/Optimizer.hpp"
#include "Activation.hpp"
#include <random>
#include <memory>

using namespace optimization;

namespace neuralnet
{
    /** Basic layer of a neural network
     *
     * @tparam F Enumerated activation function to use in this layer
     */
    template <typename T, ActivationFunc F, OptimizerClass C>
    class Layer : public Model<Layer<T, F, C>>
    {

    public:
        typedef Eigen::Vector<T, Eigen::Dynamic> InputType;
        typedef Eigen::Vector<T, Eigen::Dynamic> OutputType;

        /** Constructs a randomly initialized layer.
         *
         * Weights are initialized using He initialization, and
         * biases are initialized to 0.
         *
         * @param in_size size of the input vector to this layer
         * @param out_size size of the output vector from this layer
         */
        Layer(){ setup(0,0,0,0); }
        Layer(int in_size, int out_size, double b1, double b2){ setup(in_size, out_size, b1, b2); }
        Layer(int in_size, int out_size)
        { 
            static_assert(C==OptimizerClass::None, "Adam parameters missing"); 
            setup(in_size, out_size); 
        }
#ifndef NOPYTHON
        Layer(const py::tuple& data)
        {
            std::vector<T> w, b;
            int in = data[0].cast<int>();
            int o = data[1].cast<int>();

            if constexpr (C == OptimizerClass::Adam)
            {
                setup(in, o, data[2].cast<double>(), data[3].cast<double>());
                w = data[4].cast<std::vector<T>>();
                b = data[5].cast<std::vector<T>>();
                adam::unpickle(data[6], adam_weights);
                adam::unpickle(data[7], adam_biases);
            }
            else
            {
                setup(in, o);
                w = data[2].cast<std::vector<T>>();
                b = data[3].cast<std::vector<T>>();
            }

            weights = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(w.data(), in, o);
            biases = Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(b.data(), o);
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
         * @return (in size, out size, optimizer args..., weights, biases, (optimizer state...))
         */
        py::tuple getstate() const;
#endif

    protected:
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> weights;
        OutputType biases;
        OutputType z;
        OutputType a;
        OutputType d;
        InputType in;

        // Data for adam optimization
        adam::AdamData<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> adam_weights;
        adam::AdamData<OutputType> adam_biases;

        
        template <typename... Ts>
        void setup(Ts... Args)
        {
            auto args = std::tuple<Ts...>(Args...);

            int in_size = std::get<0>(args);
            int out_size = std::get<1>(args);

            // Apply he initialization
            this->weights = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Random(in_size, out_size).unaryExpr([in_size](double x)
                                                                                        { return x * std::sqrt(2.0 / static_cast<double>(in_size)); });

            this->biases = OutputType::Zero(out_size);
            this->z = OutputType::Zero(out_size);
            this->a = OutputType::Zero(out_size);
            this->d = OutputType::Zero(out_size);
            this->in = InputType::Zero(in_size);

            if constexpr (C == OptimizerClass::Adam)
            {
                adam_weights.m = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(in_size, out_size);
                adam_weights.v = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(in_size, out_size);
                adam_biases.m = OutputType::Zero(out_size);
                adam_biases.v = OutputType::Zero(out_size);

                adam_weights.b1 = std::get<0>(args);
                adam_weights.b2 = std::get<1>(args);
                adam_biases.b1 = std::get<0>(args);
                adam_biases.b2 = std::get<1>(args);

                adam_weights.b1powt = adam_weights.b1;
                adam_weights.b2powt = adam_weights.b2;
                adam_biases.b1powt = adam_biases.b1;
                adam_biases.b2powt = adam_biases.b2;
            }
        }
    };
}

template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
void neuralnet::Layer<T, F, C>::update(double rate)
{
    if constexpr (C == OptimizerClass::Adam)
    {
        auto tmp = in * d.transpose();
        adam::adam_update_params(rate, adam_weights, weights, tmp);
        adam::adam_update_params(rate, adam_biases, biases, d);
    }
    else
    {
        weights -= in * (d.transpose() * rate);
        biases -= rate * d;
    }
}

template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
template<typename X>
neuralnet::Layer<T, F, C>::OutputType neuralnet::Layer<T, F, C>::forward(X&& input)
{
    // Save input for this pass and calculate weighted signals
    in = {input};
    z = biases;
    z += (weights.transpose() * input);
    // Calculate and save activation function output
    a =  Activation<Eigen::Dynamic, T, F>::f(z);
    return a;
}

template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
template<typename X>
neuralnet::Layer<T, F, C>::InputType neuralnet::Layer<T, F, C>::backward(X&& err)
{
    // Calculate this layers error gradient
    d = Activation<Eigen::Dynamic, T, F>::df(z, a, err);
    // Calculate and return error gradient input to next layer
    return weights * d;
}


#ifndef NOPYTHON
template <typename T, neuralnet::ActivationFunc F, OptimizerClass C>
py::tuple neuralnet::Layer<T, F, C>::getstate() const
{
    if constexpr (C == OptimizerClass::Adam)
    {
        return py::make_tuple(
            weights.rows(), weights.cols(),
            adam_weights.b1, adam_weights.b2,
            std::vector<T>(weights.data(), weights.data() + weights.size()),
            std::vector<T>(biases.data(), biases.data() + biases.size()),
            adam::pickle(adam_weights),
            adam::pickle(adam_biases)
        );
    }
    else
    {
        return py::make_tuple(
            weights.rows(), weights.cols(),
            std::vector<T>(weights.data(), weights.data() + weights.size()),
            std::vector<T>(biases.data(), biases.data() + biases.size())
        );
    }
}
#endif


#endif
