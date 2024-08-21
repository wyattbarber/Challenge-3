#ifndef _DeepAutoEncoder_HPP
#define _DeepAutoEncoder_HPP

#include "../Model.hpp"

namespace neuralnet 
{

    template<typename T, ActivationFunc FI, ActivationFunc FH, ActivationFunc FL, optimization::OptimizerClass C>
    class DeepAutoEncoder : public Model<DeepAutoEncoder<T, FI, FH, FL, C>>
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
                latent_layer = std::make_unique<AutoEncoder<T, FL, C>>(dims[0], dims[1], Args...);
                // // Construct input layer as it has no default constructor
                // input_layer = std::make_unique<AutoEncoder<T, FI, C>>(1, 1, Args...);
            }
            else
            {
                // At least two layers exist, create input and latent
                input_layer = std::make_unique<AutoEncoder<T, FI, C>>(dims[0], dims[1], Args...);
                latent_layer = std::make_unique<AutoEncoder<T, FL, C>>(dims[n_layers-2], dims[n_layers-1], Args...);
                if(n_layers > 2)
                {
                    // Hidden layers exist, add to vector
                    for(size_t i = 1; i < n_layers-1; ++i)
                    {
                        hidden_layers.push_back(AutoEncoder<T, FH, C>(dims[i], dims[i+1], Args...));
                    }
                }
            }
        }

        template<typename X>
        OutputType forward(X&& input){ return decode(encode(input)); }

        template<typename X>
        InputType backward(X&& error){ return errorLatent(errorReconstruct(error)); }

        template<typename X>
        LatentType encode(X&& input);

        template<typename X>
        OutputType decode(X&& latent);

        template<typename X>
        LatentType errorReconstruct(X&& error);

        template<typename X>
        InputType errorLatent(X&& error);

        void update(double rate);

    protected:
        size_t n_layers;
        std::unique_ptr<AutoEncoder<T, FI, C>> input_layer;
        std::unique_ptr<AutoEncoder<T, FL, C>> latent_layer;
        std::vector<AutoEncoder<T, FH, C>> hidden_layers;
    };

    
    template<typename T, ActivationFunc FI, ActivationFunc FH, ActivationFunc FL, optimization::OptimizerClass C>
    template<typename X>
    DeepAutoEncoder<T, FI, FH, FL, C>::LatentType DeepAutoEncoder<T, FI, FH, FL, C>::encode(X&& input)
    {
        if(n_layers == 1)
        {
            return latent_layer->encode(std::forward<X>(input));
        }
        else if(n_layers == 2)
        {
            return latent_layer->encode(input_layer->encode(std::forward<X>(input)));
        }
        else
        {
            auto h = input_layer->encode(std::forward<X>(input));
            for(auto l : hidden_layers)
            {
                h = l.encode(h);
            }
            return latent_layer->encode(h);
        }
    }


    template<typename T, ActivationFunc FI, ActivationFunc FH, ActivationFunc FL, optimization::OptimizerClass C>
    template<typename X>
    DeepAutoEncoder<T, FI, FH, FL, C>::OutputType DeepAutoEncoder<T, FI, FH , FL, C>::decode(X&& latent)
    {
        if(n_layers == 1)
        {
            return latent_layer->decode(std::forward<X>(latent));
        }
        else if(n_layers == 2)
        {
            return input_layer->decode(latent_layer->decode(std::forward<X>(latent)));
        }
        else
        {
            auto h = latent_layer->decode(std::forward<X>(latent));
            for(auto l = hidden_layers.rbegin(); l != hidden_layers.rend(); ++l)
            {
                h = l->decode(h);
            }
            return input_layer->decode(h);
        }
    }


    template<typename T, ActivationFunc FI, ActivationFunc FH, ActivationFunc FL, optimization::OptimizerClass C>
    template<typename X>
    DeepAutoEncoder<T, FI, FH, FL, C>::LatentType DeepAutoEncoder<T, FI, FH , FL, C>::errorReconstruct(X&& error)
    {
        if(n_layers == 1)
        {
            return latent_layer->errorReconstruct(std::forward<X>(error));
        }
        else if(n_layers == 2)
        {
            return latent_layer->errorReconstruct(input_layer->errorReconstruct(std::forward<X>(error)));
        }
        else
        {
            auto h = input_layer->errorReconstruct(std::forward<X>(error));
            for(auto l = hidden_layers.rbegin(); l != hidden_layers.rend(); ++l)
            {
                h = l->errorReconstruct(h);
            }
            return latent_layer->errorReconstruct(h);
        }
    }


    template<typename T, ActivationFunc FI, ActivationFunc FH, ActivationFunc FL, optimization::OptimizerClass C>
    template<typename X>
    DeepAutoEncoder<T, FI, FH, FL, C>::InputType DeepAutoEncoder<T, FI, FH , FL, C>::errorLatent(X&& error)
    {
        if(n_layers == 1)
        {
            return latent_layer->errorLatent(std::forward<X>(error));
        }
        else if(n_layers == 2)
        {
            return input_layer->errorLatent(latent_layer->errorLatent(std::forward<X>(error)));
        }
        else
        {
            auto h = latent_layer->errorLatent(std::forward<X>(error));
            for(auto l : hidden_layers)
            {
                h = l.errorLatent(h);
            }
            return input_layer->errorLatent(h);
        }
    }


    template<typename T, ActivationFunc FI, ActivationFunc FH, ActivationFunc FL, optimization::OptimizerClass C>
    void DeepAutoEncoder<T, FI, FH , FL, C>::update(double rate)
    {
        latent_layer->update(rate);
        if(n_layers > 1)
        {
            input_layer->update(rate);
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