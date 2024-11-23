import pyneuralnet as nn
# Import other libraries
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
np.random.seed(123)

N = 2
a = 0.00001
AdamArgs = (0.9, 0.999)

layers = [
    nn.fullconnect.TanH(784, 500, *AdamArgs), 
    nn.fullconnect.ReLU(500, 300, *AdamArgs), 
    nn.fullconnect.ReLU(300, 100, *AdamArgs), 
    nn.fullconnect.ReLU(100, 100, *AdamArgs), 
    nn.fullconnect.ReLU(100, 50, *AdamArgs), 
    nn.fullconnect.SoftMax(50, 10, *AdamArgs)
]
model = nn.compound.Sequence(*layers)

class Data(nn.abstract.DataSource):
    def __init__(self):
        super().__init__()
        self.TRAIN_IN, self.TRAIN_OUT = pickle.load(open('data/mnist_preprocessed.pickle', 'rb'))
    
    def size(self):
        return len(self.TRAIN_IN)
    
    def sample(self, i):
        out = np.zeros((10,1))
        out[self.TRAIN_OUT[i]] = 1.0
        return (self.TRAIN_IN[i], out)

data = Data()
loss = nn.loss.L2()

trainer = nn.Trainer(model, data)

ts = time.time()
errors = trainer.train(N, a)
duration = time.time() - ts
print(f"Training of coupled autoencoder complete in {duration / N} seconds per epoch.")

plt.title("Training Error")
plt.plot(range(len(errors)), errors)
plt.show()