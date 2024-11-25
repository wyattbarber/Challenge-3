#ifndef _DYNAMIC_HPP
#define _DYNAMIC_HPP

#include "Model.hpp"


namespace neuralnet
{
    template<typename T>
    class DynamicModel
    {
        public:

        typedef T InputType; 
        typedef T OutputType; 

        virtual OutputType forward(InputType& input) = 0;   

        virtual InputType backward(OutputType& error) = 0;

        virtual void update(double rate){};
    };

    template<typename T>
    class DynamicModelTrampoline : public DynamicModel<T>
    {
        public:
            using DynamicModel<T>::DynamicModel;
            
            typedef DynamicModel<T>::InputType InputType; 
            typedef DynamicModel<T>::OutputType OutputType; 

            OutputType forward(InputType& input) override
            {
                PYBIND11_OVERRIDE_PURE(
                    OutputType, /* Return type */
                    DynamicModel<T>,      /* Parent class */
                    forward,        /* Name of function in C++ (must match Python name) */
                    input    /* Argument(s) */
                );
            }

            
            InputType backward(OutputType& error) override
            {
                PYBIND11_OVERRIDE_PURE(
                    InputType, /* Return type */
                    DynamicModel<T>,      /* Parent class */
                    backward,     /* Name of function in C++ (must match Python name) */
                    error
                );
            }

            void update(double rate) override
            {
                PYBIND11_OVERRIDE_PURE(
                    void, /* Return type */
                    DynamicModel<T>,      /* Parent class */
                    update,     /* Name of function in C++ (must match Python name) */
                    rate
                );
            }
    };


    template<class ModelType>
    class DynamicBinder : public DynamicModel<typename ModelType::InputType>
    {
        public:
        typedef DynamicModel<typename ModelType::InputType>::InputType InputType; 
        typedef DynamicModel<typename ModelType::InputType>::OutputType OutputType; 

        template<typename... Ts>
        DynamicBinder(Ts... Args) : model(Args...){}

        OutputType forward(InputType& input) override { return model.forward(input); }

        InputType backward(OutputType& error) override { return model.backward(error); }

        void update(double rate) override { model.update(rate); }


#ifndef NOPYTHON
        py::tuple getstate() const { return model.getstate(); }
#endif

        protected:
        ModelType model;
    };
}

#endif