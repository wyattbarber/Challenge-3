from . import neuralnet as nn
# Import other libraries
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from typing import List, Any
np.random.seed(123)


class Model(nn.Model2D):
    def __init__(self):
        super().__init__()
        self._pool = nn.MaxPoolEncoder2D()
        self._layer1 = nn.Conv2D(1,3, 0.9,0.999)
        # self._layer2 = nn.convolution.MaxPool2D()
        self._layer3 = nn.Conv2D(3,3, 0.9,0.999)
        # self._layer4 = nn.convolution.MaxUnPool2D(self._layer2)
        self._layer5 = nn.Conv2D(3,1, 0.9,0.999)
        self._layer6 = nn.Sigmoid2D()
    
    def forward(self, input):
        x = self._layer3.forward(
                self._pool.encode(
                    self._layer1.forward(input)
                )
            )
        return  self._layer6.forward(
                    self._layer5.forward(
                        self._pool.decode(
                            x
                        )
                    )
                )
    
    def backward(self, error):
        x = self._layer3.backward(
                            self._pool.backward_decode(
                                self._layer5.backward(
                                    self._layer6.backward(error)
                                )
                            )
                        )
        return  self._layer1.backward(
                    self._pool.backward_encode(
                        x
                    )
                )
    
    def update(self, rate):
        self._layer1.update(rate)
        self._layer3.update(rate)
        self._layer5.update(rate)
        self._layer6.update(rate)

    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, state):
        self.__dict__  = state

   

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
    

data = Data()
# model = Model()
# model = nn.MaxPoolEncoder2D()
model = nn.UNet(1, 0.6, 0.9, 0.999, True)
# model = nn.BatchRenorm2D(1, 0.8, 0.9, 0.999)

import pickle
import time

print("Running forward pass")
out1 = model.forward(data.sample(0)[0])
print(f"{np.min(out1)} - {np.mean(out1)} - {np.max(out1)}")
print(f"Pickling with format {pickle.format_version}...")
bts = pickle.dumps(model)
print("Unpickling...")
sour_model = pickle.loads(bts)
time.sleep(2.0)
print("Pickled Model...")
out2 = sour_model.forward(data.sample(0)[0])
diff = abs(out1 - out2)
print(f"{np.min(diff)} - {np.mean(diff)} - {np.max(diff)}")