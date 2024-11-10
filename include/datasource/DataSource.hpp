#ifndef _DATASOURCE_HPP
#define _DATASOURCE_HPP
#pragma once

#include <pybind11/pybind11.h>

namespace datasource{

    template<typename InputType, typename OutputType>
    class DataSource
    {
        public:
            typedef std::pair<InputType, OutputType> SampleType;

            /** Gets a single input - output pair from the dataset.
             * 
             * Must accept any i from 0 to `this->size()`, and successive calls
             * to the same index must return the same sample each time.
             * 
             * @param i Index of the sample requested. 
            */
            virtual SampleType sample(size_t i);

            /** Returns the number of samples in the dataset 
             * 
            */
            virtual size_t size() = 0;

    };

    template<typename InputType, typename OutputType>
    class DataSourceTrampoline : public DataSource<InputType, OutputType>
    {
        public:
            using DataSource<InputType, OutputType>::DataSource;

            typedef std::pair<InputType, OutputType> SampleType;
            typedef DataSource<InputType, OutputType> BaseType;

            SampleType sample(size_t i) override
            {
                PYBIND11_OVERRIDE_PURE(
                    SampleType, /* Return type */
                    BaseType,      /* Parent class */
                    sample,        /* Name of function in C++ (must match Python name) */
                    i      /* Argument(s) */
                );
            }

            
            size_t size() override
            {
                PYBIND11_OVERRIDE_PURE(
                    size_t, /* Return type */
                    BaseType,      /* Parent class */
                    size,     /* Name of function in C++ (must match Python name) */
                          
                );
            }

    };
}

#endif