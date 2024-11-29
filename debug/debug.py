from . import neuralnet as nn
# Import other libraries
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from typing import List, Any
np.random.seed(123)


class ModelUNet(nn.Model2D):
    _unet : nn.UNet
    _conv: nn.Conv2D
    _bin : nn.Sigmoid2D

    def __init__(self):
        super().__init__()
        self._unet = nn.UNet(1, 0.6, True)
        self._conv = nn.Conv2D(2, 1)
        self._bin =  nn.Sigmoid2D()
        
    def forward(self, input):
        return  self._bin.forward(
                    self._conv.forward(
                            self._unet.forward(
                                input
                            )
                    )
                )
    
    def backward(self, error):
        return  self._unet.backward(
                    self._conv.backward(
                            self._bin.backward(
                                error
                            )
                    )
                )
    
    def update(self, rate):
        self._unet.update(rate)
        self._conv.update(rate)
        self._bin.update(rate)

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
        self._train_in = [np.reshape(x, (28,28,1)).astype(np.float32) for x in train_in]
        self._train_out = [np.reshape(x, (28,28,1)).astype(np.float32) for x in train_in]

    def size(self):
        return len(self._train_in)

    def sample(self, i : int):
        return (self._train_in[i], self._train_out[i])
    
class Data1(nn.DataSource):
    _train_in : List[Any]
    _train_out : List[Any]

    def __init__(self):
        super().__init__()
        train_in, _ = pickle.load(open('test/data/mnist_preprocessed.pickle', 'rb'))
        self._train_in = [x.astype(np.float32) for x in train_in]
        self._train_out = [x.astype(np.float32) for x in train_in]

    def size(self):
        return len(self._train_in)

    def sample(self, i : int):
        return (self._train_in[i], self._train_out[i])
    

data = Data1()
# model = Model()
# model = nn.MaxPoolEncoder2D()
# model = nn.UNet(1, 0.6, 0.9, 0.999, True)
# model = nn.BatchRenorm2D(1, 0.8, 0.9, 0.999)
# model = ModelUNet()
model = nn.SigmoidAutoEncoder(28**2, 10)

import pickle
import time

print("Running forward pass")
out = model.forward(data.sample(0)[0])
print("Calculating error")
error = out - data.sample(0)[1]
print("Backpropagating")
model.backward(error)
print("Updating")
model.update(0.0001)

bts = pickle.dumps(model)
sour_model = pickle.loads(bts)

out = sour_model.forward(data.sample(0)[0])
error = out - data.sample(0)[1]
sour_model.backward(error)