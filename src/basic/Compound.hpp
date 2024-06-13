#ifndef _COMPOUND_HPP
#define _COMPOUND_HPP

#include "../Model.hpp"
#include <random>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
namespace py = pybind11;

namespace neuralnet
{
    namespace sequential
    {
        template <typename V, class M, class... Ms>
        auto forward(V &input, M &layer, Ms &...Layers)
        {
            if constexpr (sizeof...(Layers) > 0)
            {
                auto x = layer.forward(input);
                return forward(x, Layers...);
            }
            else
            {
                return layer.forward(input);
            }
        }

        template <typename V, class M, class... Ms>
        auto backward(V &error, M &layer, Ms &...Layers)
        {
            if constexpr (sizeof...(Layers) > 0)
            {
                auto x = backward(error, Layers...);
                return layer.backward(x);
            }
            else
            {
                return layer.backward(error);
            }
        }


        template <class M, class... Ms>
        void update(double rate, M &layer, Ms &...Layers)
        {
            layer.update(rate);
            if constexpr (sizeof...(Layers) > 0)
            {
                update(rate, Layers...);
            }
        }
    }

};

#endif