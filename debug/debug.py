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
        self._train_out = [np.concatenate((x,x,x), axis=2) for x in self._train_in]

    def size(self):
        return len(self._train_in)

    def sample(self, i : int):
        return (self._train_in[i], self._train_out[i])


N = 2
a = 0.001

model = nn.Conv2D(1,3)
data = Data()
trainer = nn.Trainer2D(model, data)

ts = time.time()
errors = trainer.train(N, a)
duration = time.time() - ts
print(f"Training of model complete in {duration / N} seconds per epoch.")

plt.title("Training Error")
plt.plot(range(len(errors)), errors)
plt.show()