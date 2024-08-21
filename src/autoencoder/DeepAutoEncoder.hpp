#ifndef _DeepAutoEncoder_HPP
#define _DeepAutoEncoder_HPP

#include "AutoEncoder.hpp"

namespace neuralnet 
{

    template<typename T, ActivationFunc FI, ActivationFunc FH, ActivationFunc FL, optimization::OptimizerClass C>
    class DeepAutoEncoder : public AutoEncoder
    {
    public:
        typedef Eigen::Vector<T, Eigen::Dynamic> InputType;
        typedef InputType OutputType;
        typedef Eigen::Vector<T, Eigen::Dynamic> LatentType;

        template<typename... Ts>
        DeepAutoEncoder(std::vector<size_t> dims, Ts... Args) : n_layers(dims.size() - 1)
        {
            if(n_layers == 1)
            {
                // Single layer edge case, uses latent activation function
                latent_layer = AutoEncoder<T, FL, C>(dims[0], dims[1], Args...);
            }
            else
            {
                // At least two layers exist, create input and latent
                input_layer = AutoEncoder<T, FI, C>(dims[0], dims[1], Args...);
                latent_layer = AutoEncoder<T, FL, C>(dims[n_layers-2], dims[n_layers-1], Args...);
                if(n_layers > 2)
                {
                    // Hidden layers exist, add to vector
                    for(size_t i = 1; i < n_layers-1, ++i)
                    {
                        hidden_layers.push_back(AutoEncoder<T, FH, C>(dims[i], dims[i+1], Args...));
                    }
                }
            }
        }

        OutputType forward(InputType& input){ return decode(encode(input)); }

        InputType backward(OutputType& error){ return errorLatent(errorReconstruct(error)); }

        LatentType encode(InputType& input);

        OutputType decode(LatentType& latent);

        void update(double rate);

    protected:
        size_t n_layers;
        AutoEncoder<T, FI, C> input_layer;
        AutoEncoder<T, FL, C> latent_layer;
        std::vector<AutoEncoder<T, FH, C>> hidden_layers;

        LatentType errorReconstruct(OutputType& error);

        InputType errorLatent(LatentType& error);
    };

    
    template<typename T, ActivationFunc FI, ActivationFunc FH, ActivationFunc FL, optimization::OptimizerClass C>
    Eigen::VectorXd DeepAutoEncoder<T, FI, FH, FL, C>::encode(Eigen::VectorXd& input)
    {
        if(n_layers == 1)
        {
            return latent_layer.encode(input);
        }
        else if(n_layers == 2)
        {
            return latent_layer.encode(input_layer.encode(input));
        }
        else
        {
            auto h = input_layer.encode(input);
            for(auto l : hidden_layers)
            {
                h = l.encode(h);
            }
            return latent_layer.encode(h);
        }
    }


    template<typename T, ActivationFunc FI, ActivationFunc FH, ActivationFunc FL, optimization::OptimizerClass C>
    Eigen::VectorXd DeepAutoEncoder<T, FI, FH , FL, C>::decode(Eigen::VectorXd& latent)
    {
        if(n_layers == 1)
        {
            return latent_layer.decode(latent);
        }
        else if(n_layers == 2)
        {
            return input_layer.decode(latent_layer.decode(latent));
        }
        else
        {
            auto h = latent_layer.decode(latent);
            for(auto l = hidden_layers.rbegin(); l != layers.rend(); ++l)
            {
                h = l->decode(h);
            }
            return input_layer.decode(h);
        }
    }


    template<typename T, ActivationFunc FI, ActivationFunc FH, ActivationFunc FL, optimization::OptimizerClass C>
    Eigen::VectorXd DeepAutoEncoder<T, FI, FH , FL, C>::errorReconstruct(Eigen::VectorXd& error)
    {
        if(n_layers == 1)
        {
            return latent_layer.errorReconstruct(error);
        }
        else if(n_layers == 2)
        {
            return latent_layer.errorReconstruct(input_layer.errorReconstruct(error));
        }
        else
        {
            auto h = input_layer.errorReconstruct(error);
            for(auto l = hidden_layers.rbegin(); l != layers.rend(); ++l)
            {
                h = l->errorReconstruct(h);
            }
            return latent_layer.errorReconstruct(h);
        }
    }


    template<typename T, ActivationFunc FI, ActivationFunc FH, ActivationFunc FL, optimization::OptimizerClass C>
    Eigen::VectorXd DeepAutoEncoder<T, FI, FH , FL, C>::errorLatent(Eigen::VectorXd& error)
    {
        if(n_layers == 1)
        {
            return latent_layer.errorLatent(error);
        }
        else if(n_layers == 2)
        {
            return input_layer.errorLatent(latent_layer.errorLatent(error));
        }
        else
        {
            auto h = latent_layer.errorLatent(error);
            for(auto l : layers)
            {
                h = l.errorLatent(h);
            }
            return input_layer.errorLatent(h);
        }
    }


    template<typename T, ActivationFunc FI, ActivationFunc FH, ActivationFunc FL, optimization::OptimizerClass C>
    void DeepAutoEncoder<T, FI, FH , FL, C>::update(double rate)
    {
        latent_layer.update(rate);
        if(n_layers > 1)
        {
            input_layer.update(rate);
        }
        if(n_layers > 2)
        {
            for(auto layer : hidden_layers)
            {
                layer.update(rate);
            }
        }
    }

}

#endif