from . import neuralnet as nn
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
np.random.seed(123)


class Data(nn.DataSource):
    def __init__(self):
        super().__init__()
        self.TRAIN_IN, self.TRAIN_OUT = pickle.load(open('test/data/mnist_preprocessed.pickle', 'rb'))
    
    def size(self):
        return len(self.TRAIN_IN)
    
    def sample(self, i):
        return (self.TRAIN_IN[i], self.TRAIN_IN[i])


data = Data()
N = 20
a = 0.0001
AdamArgs = (0.9, 0.999)

model = nn.AutoEncoder(784, 50, *AdamArgs)
trainer = nn.Trainer(model, data)

ts = time.time()
errors = trainer.train(N, a)
duration = time.time() - ts
print(f"Training of coupled autoencoder complete in {duration / N} seconds per epoch.")

plt.title("Training Error")
plt.plot(range(len(errors)), errors)
plt.show()