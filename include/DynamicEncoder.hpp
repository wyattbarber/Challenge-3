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
        
        typedef T::Scalar Scalar;
        typedef T InputType; 
        typedef T OutputType; 
        typedef V LatentType; 

        virtual LatentType encode(InputType& input) = 0;   

        virtual OutputType decode(LatentType& embed) = 0;

        virtual InputType backward_encode(LatentType& error) = 0;   

        virtual LatentType backward_decode(OutputType& error) = 0;
    };

    template<typename T, typename V>
    class DynamicEncoderTrampoline : public DynamicEncoder<T,V>
    {
        public:
            using DynamicEncoder<T,V>::DynamicEncoder;
            
            typedef DynamicEncoder<T,V>::Scalar Scalar;
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

            InputType backward_encode(LatentType& error) override
            {
                PYBIND11_OVERRIDE_PURE(
                    InputType, /* Return type */
                    BaseType,      /* Parent class */
                    backward_encode,        /* Name of function in C++ (must match Python name) */
                    error    /* Argument(s) */
                );
            }
            
            LatentType backward_decode(OutputType& error) override
            {
                PYBIND11_OVERRIDE_PURE(
                    LatentType, /* Return type */
                    BaseType,      /* Parent class */
                    backward_decode,     /* Name of function in C++ (must match Python name) */
                    error
                );
            }
    };


    template<class ModelType>
    class DynamicEncoderBinder : public DynamicEncoder<typename ModelType::InputType, typename ModelType::LatentType>
    {
        public:
        typedef DynamicEncoder<typename ModelType::InputType, typename ModelType::LatentType>::Scalar Scalar; 
        typedef DynamicEncoder<typename ModelType::InputType, typename ModelType::LatentType>::InputType InputType; 
        typedef DynamicEncoder<typename ModelType::InputType, typename ModelType::LatentType>::OutputType OutputType; 
        typedef DynamicEncoder<typename ModelType::InputType, typename ModelType::LatentType>::LatentType LatentType; 

        template<typename... Ts>
        DynamicEncoderBinder(Ts... Args) : model(Args...){}

        OutputType forward(InputType& input) override { return model.forward(input); }

        InputType backward(OutputType& error) override { return model.backward(error); }
        
        LatentType encode(InputType& input) override { return model.encode(input); }

        OutputType decode(LatentType& embed) override { return model.decode(embed); }

        InputType backward_encode(LatentType& error) override { return model.backward_encode(error); }

        LatentType backward_decode(OutputType& error) override { return model.backward_decode(error); }

        void update(double rate) override { model.update(rate); }

#ifndef NOPYTHON
        py::tuple getstate() const { return model.getstate(); }
#endif

        protected:
        ModelType model;
    };
}

#endif