#ifndef _UNET_HPP
#define _UNET_HPP

#include "Conv2D.hpp"
#include "Activation2D.hpp"
#include "PoolUnPool2D.hpp"
#include "../normalize/ReNorm2D.hpp"

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
     *  * 2N-to-2N channel 3x3 convolution
     *  * Normalize and ReLU, store intermediate
     *  * 2x2 max pool
     * 
     * decode:
     *  * 2x2 up convert
     *  * 4N-to-2N channel 3x3 convolution
     *  * Normalize
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
    template <typename T, template<typename> class C>
    class UNet : public Encoder<UNet<T, C>>
    {   
        public:
            typedef T Scalar;
            typedef Convolution2D<T, 3, C>::InputType InputType;
            typedef Layer2D<T, ActivationFunc::ReLU>::OutputType OutputType;
            typedef Layer2D<T, ActivationFunc::ReLU>::OutputType LatentType;

            /** Constructs a UNet component. 
             * 
             * If this component is the innermost part of the model
             * (its encoder output will be fed directly to its decoder),
             * then `final` should be specified as true. This will add additional
             * convolutions so that the models encode generates 4N channels rather than 2N.
             * 
             * @param N number of input channels.
             * @param alpha batch renormalization update rate.
             * @param final this is the final (innermost) component.
            */
            UNet(int N, T alpha, bool final = false) :
                is_final(final),
                relu_enc_1(),
                relu_enc_2(),
                relu_dec_2(),
                relu_dec_3(),
                conv_enc_1(N, 2*N),
                norm_enc_1(2*N, alpha),
                conv_enc_2(2*N, 2*N),
                norm_enc_2(2*N, alpha),
                conv_dec_1(4*N, 2*N),
                norm_dec_1(2*N, alpha),
                conv_dec_2(4*N, 2*N),
                norm_dec_2(2*N, alpha),
                conv_dec_3(2*N, 2*N),
                norm_dec_3(2*N, alpha),
                pool(),
                unpool()
            {
                if(is_final)
                {
                    conv_enc_3 = std::make_unique<Convolution2D<T, 3, C>>(2*N, 4*N);
                    relu_enc_3 = std::make_unique<Layer2D<T, ActivationFunc::ReLU>>();
                    norm_enc_3 = std::make_unique<ReNorm2D<T,C>>(4*N, alpha);
                    conv_enc_4 = std::make_unique<Convolution2D<T, 3, C>>(4*N, 4*N);
                    relu_enc_4 = std::make_unique<Layer2D<T, ActivationFunc::ReLU>>();
                    norm_enc_4 = std::make_unique<ReNorm2D<T,C>>(4*N, alpha);
                }
            }

#ifndef NOPYTHON 
            /** Unpickling constructor
             * 
             */
            UNet(const py::tuple& data) :
                is_final(data[0].cast<bool>()),
                relu_enc_1(),
                relu_enc_2(),
                relu_dec_2(),
                relu_dec_3(),
                conv_enc_1(data[1]),
                norm_enc_1(data[2]),
                conv_enc_2(data[3]),
                norm_enc_2(data[4]),
                conv_dec_1(data[5]),
                norm_dec_1(data[6]),
                conv_dec_2(data[7]),
                norm_dec_2(data[8]),
                conv_dec_3(data[9]),
                norm_dec_3(data[10]),
                pool(),
                unpool()
            {
                if(is_final)
                {
                    conv_enc_3 = std::make_unique<Convolution2D<T, 3, C>>(data[11]);
                    relu_enc_3 = std::make_unique<Layer2D<T, ActivationFunc::ReLU>>();
                    norm_enc_3 = std::make_unique<ReNorm2D<T,C>>(data[12]);
                    conv_enc_4 = std::make_unique<Convolution2D<T, 3, C>>(data[13]);
                    relu_enc_4 = std::make_unique<Layer2D<T, ActivationFunc::ReLU>>();
                    norm_enc_4 = std::make_unique<ReNorm2D<T,C>>(data[14]);
                }
            }
#endif
            template<typename X>
            OutputType forward(X&& input){ return decode(encode(std::forward<X>(input))); }

            template<typename X>
            InputType backward(X&& error){ return backward_encode(backward_decode(std::forward<X>(error))); }

            template<typename X>
            LatentType encode(X&& input)
            {
                auto a = relu_enc_1.forward(
                                norm_enc_1.forward(
                                    conv_enc_1.forward(
                                        std::forward<X>(input)
                                    )
                                )
                            );
                inter = relu_enc_2.forward(
                    norm_enc_2.forward(
                        conv_enc_2.forward(
                            a
                        )   
                    )
                );
                
                return is_final ? 
                    relu_enc_4->forward(
                        norm_enc_4->forward(
                            conv_enc_4->forward(
                                relu_enc_3->forward(
                                    norm_enc_3->forward(
                                        conv_enc_3->forward(
                                            pool.forward( 
                                                inter
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                    : 
                    pool.forward(inter)
                ;
            }

            template<typename X>
            OutputType decode(X&& embed)
            {
                // Upsample
                auto tmp =  norm_dec_1.forward(
                                conv_dec_1.forward(
                                    unpool.forward(
                                        std::forward<X>(embed)
                                    )
                                )
                            );

                // Concatenate and continue
                return relu_dec_3.forward(
                    norm_dec_3.forward(
                        conv_dec_3.forward(
                            relu_dec_2.forward(
                                norm_dec_2.forward(
                                    conv_dec_2.forward(
                                        static_cast<Eigen::Tensor<T,3>>(tmp.concatenate(inter,2))
                                    )
                                )
                            )
                        )
                    )
                );
            }

            template<typename X>
            InputType backward_encode(X&& error)
            {
                // backpropagate downsampling
                auto tmp = pool.backward( 
                    is_final ? 
                    conv_enc_3->backward(
                        norm_enc_3->backward(
                            relu_enc_3->backward(
                                conv_enc_4->backward(
                                    norm_enc_4->backward(
                                        relu_enc_4->backward(
                                            std::forward<X>(error)
                                        )
                                    )
                                )
                            )
                        )
                    )
                    :
                    std::forward<X>(error)
                );
                // complete backpropagation with combined gradient
                return conv_enc_1.backward(
                    norm_enc_1.backward(
                        relu_enc_1.backward(
                            conv_enc_2.backward(
                                norm_enc_2.backward(
                                    relu_enc_2.backward(
                                        static_cast<Eigen::Tensor<T,3>>(tmp + inter)
                                    )
                                )
                            )
                        )
                    )
                );
            }

            template<typename X>
            LatentType backward_decode(X&& error)
            {
                // Backpropagate upsampled data
                auto tmp = conv_dec_2.backward(
                    norm_dec_2.backward(
                        relu_dec_2.backward(
                            conv_dec_3.backward(
                                norm_dec_3.backward(
                                    relu_dec_3.backward(
                                        std::forward<X>(error)
                                    )
                                )
                            )
                        )
                    )
                );

                // Split 4N channel gradients to 2N downconversion gradient and 2N high res gradient
                Eigen::array<Eigen::Index,3> slice_size = {tmp.dimension(0), tmp.dimension(1), tmp.dimension(2)/2};
                Eigen::array<Eigen::Index,3> slice_tmp_start = {0, 0, 0};
                Eigen::array<Eigen::Index,3> slice_inter_start = {0, 0, tmp.dimension(2)/2};
                inter = tmp.slice(slice_inter_start, slice_size);

                // Complete backpropagation over upsampler
                return  unpool.backward(
                            conv_dec_1.backward(
                                norm_dec_1.backward(static_cast<Eigen::Tensor<T,3>>(tmp.slice(slice_tmp_start, slice_size)))
                            )
                        );
            }

            void update(double rate)
            {
                conv_enc_1.update(rate);
                norm_enc_1.update(rate);
                conv_enc_2.update(rate);
                norm_enc_2.update(rate);
                conv_dec_1.update(rate);
                norm_dec_1.update(rate);
                conv_dec_2.update(rate);
                norm_dec_2.update(rate);
                conv_dec_3.update(rate);
                norm_dec_3.update(rate);
                if(is_final)
                {
                    conv_enc_3->update(rate);
                    norm_enc_3->update(rate);
                    conv_enc_4->update(rate);
                    norm_enc_4->update(rate);
                }
            }
        
#ifndef NOPYTHON
            /** Pickling implementation
             * 
             * Returns following data in order:
             * * Final layer flag
             * * Encoder convolution 1 state
             * * Encoder renorm 1 state
             * * Encoder convolution 2 state
             * * Encoder renorm 2 
             * * Decoder convolution 1 state
             * * Decoder renorm 1 state
             * * Decoder convolution 2 state
             * * Decoder renorm 2 state
             * * Decoder convolution 3 state
             * * Decoder renorm 3 state
             * * Extra encoder convolution 1 state, if final layer flag is set
             * * Extra encoder norm 1 state, if final layer flag is set
             * * Extra encoder convolution 2 state, if final layer flag is set
             * * Extra encoder norm 2 state, if final layer flag is set
             * 
             * 
             * @return (in channels, out channels, optimizer args..., kernels, biases)
             */
            py::tuple getstate() const { 
                return py::make_tuple(
                    is_final,
                    conv_enc_1.getstate(),
                    norm_enc_1.getstate(),
                    conv_enc_2.getstate(),
                    norm_enc_2.getstate(),
                    conv_dec_1.getstate(),
                    norm_dec_1.getstate(),
                    conv_dec_2.getstate(),
                    norm_dec_2.getstate(),
                    conv_dec_3.getstate(),
                    norm_dec_3.getstate(),
                    is_final ? conv_enc_3->getstate() : py::tuple(),
                    is_final ? norm_enc_3->getstate() : py::tuple(),
                    is_final ? conv_enc_4->getstate() : py::tuple(),
                    is_final ? norm_enc_4->getstate() : py::tuple()
                ); 
            }
#endif
            /** Switch training or evaluation mode
             * 
             */
            void _mode(bool mode)
            {
                if(mode)
                {
                    conv_enc_1.train();
                    norm_enc_1.train();
                    relu_enc_1.train();
                    conv_enc_2.train();
                    norm_enc_2.train();
                    relu_enc_2.train();
                    conv_dec_1.train();
                    norm_dec_1.train();
                    conv_dec_2.train();
                    norm_dec_2.train();
                    relu_dec_2.train();
                    conv_dec_3.train();
                    norm_dec_3.train();
                    relu_dec_3.train();
                    if(is_final)
                    {
                        conv_enc_3->train();
                        norm_enc_3->train();
                        relu_enc_3->train();
                        conv_enc_4->train();
                        norm_enc_4->train();
                        relu_enc_4->train();
                    }
                }
                else
                {
                    conv_enc_1.eval();
                    norm_enc_1.eval();
                    relu_enc_1.eval();
                    conv_enc_2.eval();
                    norm_enc_2.eval();
                    relu_enc_2.eval();
                    conv_dec_1.eval();
                    norm_dec_1.eval();
                    conv_dec_2.eval();
                    norm_dec_2.eval();
                    relu_dec_2.eval();
                    conv_dec_3.eval();
                    norm_dec_3.eval();
                    relu_dec_3.eval();
                    if(is_final)
                    {
                        conv_enc_3->eval();
                        norm_enc_3->eval();
                        relu_enc_3->eval();
                        conv_enc_4->eval();
                        norm_enc_4->eval();
                        relu_enc_4->eval();
                    }
                }
            }

        protected:
            const bool is_final; 

            Layer2D<T, ActivationFunc::ReLU> relu_enc_1, relu_enc_2, relu_dec_2, relu_dec_3;

            Convolution2D<T, 3, C> conv_enc_1, conv_enc_2;
            ReNorm2D<T,C> norm_enc_1, norm_enc_2, norm_dec_1, norm_dec_2, norm_dec_3;
            Convolution2D<T, 3, C> conv_dec_1, conv_dec_2, conv_dec_3;
            
            // Extra layers for final model
            std::unique_ptr<Convolution2D<T, 3, C>> conv_enc_3, conv_enc_4;
            std::unique_ptr<ReNorm2D<T,C>> norm_enc_3, norm_enc_4;
            std::unique_ptr<Layer2D<T, ActivationFunc::ReLU>> relu_enc_3, relu_enc_4;

            Pool2D<T,2,PoolMode::Max> pool;
            UnPool2D<T,2,PoolMode::Mean> unpool;
            // Intermediate data shared between encode and decode. Used for both forward data and gradients.
            Eigen::Tensor<T,3> inter;
    };
}

#endif