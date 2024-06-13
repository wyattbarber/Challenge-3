#ifndef _CONVERTER_HPP
#define _CONVERTER_HPP

#include "../Model.hpp"


namespace neuralnet
{
    namespace conversions
    {
    template <int I, typename T>
    class Static2Dynamic : public Model<I, Eigen::Dynamic, T>
    {
    public:
        Eigen::Vector<T, Eigen::Dynamic> forward(Eigen::Vector<T, I> &input){ return static_cast<Eigen::Vector<T, Eigen::Dynamic>>(input); }

        Eigen::Vector<T, I> backward(Eigen::Vector<T, Eigen::Dynamic> &error){ return static_cast<Eigen::Vector<T, I>>(error); }

        void update(double rate){}
    };

    template <int O, typename T>
    class Dynamic2Static : public Model<Eigen::Dynamic, O, T>
    {
    public:
        Eigen::Vector<T, O> forward(Eigen::Vector<T, Eigen::Dynamic> &input){ return static_cast<Eigen::Vector<T, O>>(input); }

        Eigen::Vector<T, Eigen::Dynamic> backward(Eigen::Vector<T, O> &error){ return static_cast<Eigen::Vector<T, Eigen::Dynamic>>(error); }

        void update(double rate){}
    };


    }
}

#endif