#ifndef _RESHAPE_HPP
#define _RESHAPE_HPP

#include "../Model.hpp"


namespace neuralnet {

    /** Converts 2D tensor data to 1D vector data 
     * 
     * Input is 3D tensor data (2D spatial, 1 dimension of channels),
     * and output is the same data flattened. The backpropagation
     * reverses this transformation.
    */
    template <typename T>
    class Reshape1D : public Model<Reshape1D<T>>
    {
        public:
        typedef Eigen::Tensor<T,3> InputType;
        typedef Eigen::Vector<T,Eigen::Dynamic> OutputType;
        
        template<typename X>
        OutputType forward(X&& input)
        {
            Eigen::Tensor<T, 1> flat = input.reshape(Eigen::array<Eigen::Index, 1>{input.size()});
            dims = input.dimensions();
            return Eigen::Map<Eigen::VectorXd>(flat.data(), flat.size());
        }

        template<typename X>
        InputType backward(X&& error)
        {
            Eigen::TensorMap<Eigen::Tensor<T,1>> im(error.data(), error.size());
            return im.reshape(dims);
        }

        void update(double rate){};

#ifndef NOPYTHON
        /** Pickling implementation
         * 
         * This model has no state or constructor args,
         * so it is only defined for compatibility and 
         * to allow it to be part of larger picklable models.
         *  
         * @return empty
         */
        static py::tuple getstate(const Reshape1D<T>& obj){ return py::tuple(); }

        static Reshape1D<T> setstate(py::tuple data){ return Reshape1D<T>(); }
#endif

        protected:
        InputType::Dimensions dims; /// Size and shape of last input tensor
    };
}


#endif