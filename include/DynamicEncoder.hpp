#ifndef _DYNAMIC_ENCODER_HPP
#define _DYNAMIC_ENCODER_HPP

#include "Model.hpp"
#include "Dynamic.hpp"


namespace neuralnet
{
    template<typename T, typename V>
    class DynamicEncoder : public DynamicModel<T>
    {
        public:

        typedef T InputType; 
        typedef T OutputType; 
        typedef V LatentType; 

        virtual LatentType encode(InputType& input) = 0;   

        virtual OutputType decode(LatentType& error) = 0;
    };

    template<typename T, typename V>
    class DynamicEncoderTrampoline : public DynamicEncoder<T,V>
    {
        public:
            using DynamicEncoder<T,V>::DynamicEncoder;
            
            typedef DynamicEncoder<T,V>::InputType InputType; 
            typedef DynamicEncoder<T,V>::OutputType OutputType; 
            typedef DynamicEncoder<T,V>::LatentType LatentType; 

            typedef DynamicEncoder<T,V> BaseType;

            OutputType forward(InputType& input) override
            {
                PYBIND11_OVERRIDE_PURE(
                    OutputType, /* Return type */
                    BaseType,      /* Parent class */
                    forward,        /* Name of function in C++ (must match Python name) */
                    input    /* Argument(s) */
                );
            }

            
            InputType backward(OutputType& error) override
            {
                PYBIND11_OVERRIDE_PURE(
                    InputType, /* Return type */
                    BaseType,      /* Parent class */
                    backward,     /* Name of function in C++ (must match Python name) */
                    error
                );
            }

            void update(double rate) override
            {
                PYBIND11_OVERRIDE_PURE(
                    void, /* Return type */
                    BaseType,      /* Parent class */
                    update,     /* Name of function in C++ (must match Python name) */
                    rate
                );
            }

            LatentType encode(InputType& input) override
            {
                PYBIND11_OVERRIDE_PURE(
                    LatentType, /* Return type */
                    BaseType,      /* Parent class */
                    encode,        /* Name of function in C++ (must match Python name) */
                    input    /* Argument(s) */
                );
            }
            
            OutputType decode(LatentType& embed) override
            {
                PYBIND11_OVERRIDE_PURE(
                    OutputType, /* Return type */
                    BaseType,      /* Parent class */
                    decode,     /* Name of function in C++ (must match Python name) */
                    embed
                );
            }
    };


    template<class ModelType>
    class DynamicEncoderBinder : public DynamicEncoder<typename ModelType::InputType, typename ModelType::LatentType>
    {
        public:
        typedef DynamicEncoder<typename ModelType::InputType, typename ModelType::LatentType>::InputType InputType; 
        typedef DynamicEncoder<typename ModelType::InputType, typename ModelType::LatentType>::OutputType OutputType; 
        typedef DynamicEncoder<typename ModelType::InputType, typename ModelType::LatentType>::LatentType LatentType; 

        template<typename... Ts>
        DynamicEncoderBinder(Ts... Args) : model(Args...){}
        DynamicEncoderBinder(ModelType m){ model = m; }

        OutputType forward(InputType& input) override { return model.forward(input); }

        InputType backward(OutputType& error) override { return model.backward(error); }
        
        LatentType encode(InputType& input) override { return model.encode(input); }

        OutputType decode(LatentType& embed) override { return model.decode(embed); }

        void update(double rate) override { model.update(rate); }

        const ModelType* getmodel() const { return &model; }

        protected:
        ModelType model;
    };
}

#endif