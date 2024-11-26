#ifndef _UNET_HPP
#define _UNET_HPP

#include "Conv2D.hpp"
#include "Activation2D.hpp"
#include "PoolUnPool2D.hpp"

using namespace optimization;

namespace neuralnet {
    /** Basic sub-component of the U-Net architecture
     * 
     * Implements one pool/unpool layer of the U-Net model architecture,
     * consisting of the following encode and decode phases:
     * 
     * encode: 
     *  * N-to-2N channel 3x3 convolution
     *  * Normalize and ReLU
     *  * 2N-to-2N channel 3x3 convolution and store output
     *  * Normalize and ReLU
     *  * 2x2 max pool
     * 
     * decode:
     *  * 4N-to-2N channel 3x3 convolution
     *  * 2x2 up convert
     *  * Concatenate with stored encoder intermediate
     *  * 4N-to-2N channel 3x3 convolution
     *  * Normalize and ReLU
     *  * 2N-to-2N channel 3x3 convolution
     *  * Normalize and ReLU
     * 
     * If the `final` flag is given to the constructor, then the following 
     * layers are added to the encode step to match the latent dimensions.
     * Otherwise, forward and backward should not be called directly.
     * 
     *  * 2N-to-4N channel 3x3 convolution
     *  * Normalize and ReLU
     *  * 4N-to-4N channel 3x3 convolution
     *  * Normalize and ReLU
     */
    template <typename T, OptimizerClass C>
    class UNet : public Encoder<UNet<T, C>>
    {   
        public:
            typedef Eigen::Tensor<T, 3> InputType;
            typedef Eigen::Tensor<T, 3> OutputType;
            typedef Eigen::Tensor<T, 3> LatentType;

            /** Constructs a UNet component. 
             * 
             * If this component is the innermost part of the model
             * (its encoder output will be fed directly to its decoder),
             * then `final` should be specified as true. This will add additional
             * convolutions so that the models encode generates 4N channels rather than 2N.
             * 
             * @param N number of input channels
             * @param final this is the final (innermost) component
            */
            UNet(int N, bool final = false) :
                relu(),
                conv_enc_1(N, 2*N),
                conv_enc_2(2*N, N),
                conv_dec_1(4*N, 2*N),
                conv_dec_2(4*N, 2*N),
                conv_dec_3(2*N, 2*N),
                pool(),
                unpool()
            { this->is_final = final; }
            UNet(int N, T b1, T b2,  bool final = false) :
                relu(),
                conv_enc_1(N, 2*N, b1, b2),
                conv_enc_2(2*N, N, b1, b2),
                conv_dec_1(4*N, 2*N, b1, b2),
                conv_dec_2(4*N, 2*N, b1, b2),
                conv_dec_3(2*N, 2*N, b1, b2),
                pool(),
                unpool()
            { this->is_final = final; }

#ifndef NOPYTHON 
            /** Unpickling constructor
             * 
             */
            UNet(py::tuple data) :
                relu(),
                conv_enc_1(data[1]),
                conv_enc_2(data[2]),
                conv_dec_1(data[3]),
                conv_dec_2(data[4]),
                conv_dec_3(data[5]),
                pool(),
                unpool()
            {   
                this->is_final = data[0].cast<bool>();
            }
#endif
            template<typename X>
            OutputType forward(X&& input){ return decode(encode(std::forward<X>(input))); }

            template<typename X>
            InputType backward(X&& error){ return backward_encode(backward_decode(std::forward<X>(error))); }

            template<typename X>
            LatentType encode(X&& input)
            {
                return pool.forward(
                    conv_enc_2.forward(
                        conv_enc_1.forward(
                            std::forward<X>(input)
                        )
                    )
                );
            }

            template<typename X>
            OutputType decode(X&& embed)
            {
                return conv_dec_3.forward(
                    conv_dec_2.forward(
                        unpool.forward(
                            conv_dec_1.forward(
                                std::forward<X>(embed)
                            )
                        )
                    )
                );
            }

            template<typename X>
            InputType backward_encode(X&& error)
            {
                return conv_enc_1.backward(
                    conv_enc_2.backward(
                        pool.backward(
                            std::forward<X>(error)
                        )
                    )
                );
            }

            template<typename X>
            LatentType backward_decode(X&& error)
            {
                return conv_dec_1.backward(
                    unpool.backward(
                        conv_dec_2.backward(
                            conv_dec_3.backward(
                                std::forward<X>(error)
                            )
                        )
                    )
                );
            }

            void update(double rate)
            {
                conv_enc_1.update(rate);
                conv_enc_2.update(rate);
                conv_dec_1.update(rate);
                conv_dec_2.update(rate);
                conv_dec_3.update(rate);
            }
        
#ifndef NOPYTHON
            /** Pickling implementation
             * 
                conv_enc_1(data[1]),
                conv_enc_2(data[2]),
                conv_dec_1(data[3]),
                conv_dec_2(data[4]),
                conv_dec_3(data[5]),
             *  
             * @return (in channels, out channels, optimizer args..., kernels, biases)
             */
            py::tuple getstate() const { 
                return py::make_tuple(
                    is_final,
                    conv_enc_1.getstate(),
                    conv_enc_2.getstate(),
                    conv_dec_1.getstate(),
                    conv_dec_2.getstate(),
                    conv_dec_3.getstate()
                ); 
            }
#endif

        protected:
            bool is_final; 
            Activation2D<T, ActivationFunc::ReLU> relu; // 2D ReLU has no state, so one instance can be shared
            Convolution2D<T, 3, C> conv_enc_1, conv_enc_2;
            Convolution2D<T, 3, C> conv_dec_1;
            Convolution2D<T, 3, C> conv_dec_2, conv_dec_3;
            Pool2D<T,2,PoolMode::Max> pool;
            UnPool2D<T,2,PoolMode::Mean> unpool;

            Eigen::Tensor<T,3> inter, grad_inter; // Intermediate value concatenated during decoding and its gradient
    };
}

#endif