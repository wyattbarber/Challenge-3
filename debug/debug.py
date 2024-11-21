from . import neuralnet as nn
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from typing import List, Any
np.random.seed(123)


class Data(nn.DataSource2D):
    _train_in : List[Any]
    _train_out : List[Any]

    def __init__(self):
        super().__init__()
        train_in, _ = pickle.load(open('test/data/mnist_preprocessed.pickle', 'rb'))
        self._train_in = [np.reshape(x, (28,28,1)) for x in train_in]
        self._train_out = [np.reshape(x, (28,28,1)) for x in train_in]

    def size(self):
        return len(self._train_in)

    def sample(self, i : int):
        return (self._train_in[i], self._train_out[i])


class Model(nn.Model2D):
    def __init__(self):
        super().__init__()
        self._layer1 = nn.Conv2D(1,3, 0.9,0.999)
        self._layer2 = nn.MaxPool2D()
        self._layer3 = nn.Conv2D(3,3, 0.9,0.999)
        self._layer4 = nn.MaxUnPool2D(self._layer2)
        self._layer5 = nn.Conv2D(3,1, 0.9,0.999)
        self._layer6 = nn.Sigmoid2D()
    
    def forward(self, input):
        return  self._layer6.forward(
                    self._layer5.forward(
                        self._layer4.forward(
                            self._layer3.forward(
                                self._layer2.forward(
                                    self._layer1.forward(input)
                                )
                            )
                        )
                    )
                )
    
    def backward(self, error):
        return  self._layer1.backward(
                    self._layer2.backward(
                        self._layer3.backward(
                            self._layer4.backward(
                                self._layer5.backward(
                                    self._layer6.backward(error)
                                )
                            )
                        )
                    )
                )
    
    def update(self, rate):
        self._layer1.update(rate)
        self._layer2.update(rate)
        self._layer3.update(rate)
        self._layer4.update(rate)
        self._layer5.update(rate)
        self._layer6.update(rate)


# layer1 = nn.Conv2D(1,3, 0.9,0.999)
# layer2 = nn.MaxPool2D()
# layer3 = nn.Conv2D(3,3, 0.9,0.999)
# layer4 = nn.MaxUnPool2D(layer2)
# layer5 = nn.Conv2D(3,1, 0.9,0.999)
# layer6 = nn.Sigmoid2D()


data = Data()
N = 2
a = 0.0001

model = Model()
# model = nn.Sequence2D([layer1, layer3, layer5, layer6])
trainer = nn.Trainer2D(model, data)

ts = time.time()
errors = trainer.train(N, a)
duration = time.time() - ts
print(f"Training of coupled autoencoder complete in {duration / N} seconds per epoch.")

plt.title("Training Error")
plt.plot(range(len(errors)), errors)
plt.show()