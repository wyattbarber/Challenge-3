from . import neuralnet as nn
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import sys
from typing import List, Any

np.random.seed(123)



class Data(nn.DataSource2D):
    _train_in : List[Any]
    _train_out : List[Any]

    def __init__(self):
        super().__init__()
        train_in, _ = pickle.load(open('test/data/mnist_preprocessed.pickle', 'rb'))
        self._train_in = [np.reshape(x, (28,28,1)) for x in train_in]
        train_out = [np.concatenate((x,x,x), axis=2) for x in self._train_in]
        self._train_out = [x[::5, ::5, :] for x in train_out]

    def size(self):
        return len(self._train_in)

    def sample(self, i : int):
        return (self._train_in[i], self._train_out[i])

class Model(nn.Conv2D):
    def __init__(self):
        super().__init__(1,3)
        self._layer2 = nn.MaxPool2D()
    
    def forward(self, input):
        out = self._layer2.forward(super().forward(input))
        return out
    
    def backward(self, error):
        return super().backward(self._layer2.backward(error))
    
    def update(self, rate):
        super().update(rate)
        self._layer2.update(rate)

N = 2
a = 0.001

model = Model()
data = Data()
trainer = nn.Trainer2D(model, data)

print(model.forward(data.sample(0)[0]).shape)

ts = time.time()
errors = trainer.train(N, a)
duration = time.time() - ts
print(f"Training of model complete in {duration / N} seconds per epoch.")

plt.title("Training Error")
plt.plot(range(len(errors)), errors)
plt.show()