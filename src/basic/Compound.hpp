#ifndef _COMPOUND_HPP
#define _COMPOUND_HPP

#include "../Model.hpp"
#include <random>
#include <vector>

namespace neuralnet
{
    namespace sequential
    {
        template <typename V, class M, class... Ms>
        auto forward(V&& input, M &layer, Ms &...Layers)
        {
            if constexpr (sizeof...(Layers) > 0)
            {
                return forward(layer.forward(std::forward<V>(input)), Layers...);
            }
            else
            {
                return layer.forward(std::forward<V>(input));
            }
        }

        template <typename V, class M, class... Ms>
        auto backward(V&& error, M &layer, Ms &...Layers)
        {
            if constexpr (sizeof...(Layers) > 0)
            {
                return layer.backward(backward(std::forward<V>(error), Layers...));
            }
            else
            {
                return layer.backward(std::forward<V>(error));
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