

Layer.update

        /**
         * Updates parameters of this layer based on the previously propagated error, using the Adam algorithm
         * @param rate learning rate
         * @param b1 first moment decay rate
         * @param b2 second moment decay rate
         * @param t current training step
         */
        void update(double rate, double b1, double b2, int t)
        {
            Eigen::VectorXd mhat, vhat;

            for (int n = 0; n < d.size(); ++n)
            {
                Eigen::VectorXd grad = d(n) * in;
                m.col(n) = (b1 * m.col(n)) + ((1.0 - b1) * grad);
                v.col(n) = (b2 * v.col(n)) + ((1.0 - b2) * (grad.cwiseProduct(grad)));
                mhat = m / (1.0 - std::pow(b1, static_cast<double>(t)));
                vhat = v / (1.0 - std::pow(b2, static_cast<double>(t)));

                for (int i = 0; i < mhat.size(); ++i)
                {
                    weights.col(n)(i) -= rate * mhat(i) / std::sqrt(vhat(i) + 0.000001);
                }
            }
            mb = (b1 * mb) + ((1.0 - b1) * d);
            vb = (b2 * vb) + ((1.0 - b2) * (d.cwiseProduct(d)));
            mhat = mb / (1.0 - std::pow(b1, static_cast<double>(t)));
            vhat = vb / (1.0 - std::pow(b2, static_cast<double>(t)));
            for (int i = 0; i < mhat.size(); ++i)
            {
                biases(i) -= rate * mhat(i) / std::sqrt(vhat(i) + 0.000001);
            }
        };


Sequence.train


/**
 * Runs a backpropagation epoch through the model using the Adam algorithm
 *
 * @param inputs list of N input vectors to train on
 * @param outputs list of N correct output vectors
 * @param rate learning rate
 * @param passes number of times to pass over the input data
 *
 * @return list containing the error of each epoch
 */
std::vector<double> neuralnet::Sequence::train(std::vector<Eigen::VectorXd> inputs, std::vector<Eigen::VectorXd> outputs, double rate, int passes, double b1, double b2)
{

}

