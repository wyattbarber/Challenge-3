from . import neuralnet as nn
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import sys

np.random.seed(123)

TRAIN_IN, _ = pickle.load(open('data/mnist_preprocessed.pickle', 'rb'))
TRAIN_IN = [np.reshape(x, (28,28,1)) for x in TRAIN_IN]
TRAIN_OUT = [np.concatenate((x,x,x), axis=2) for x in TRAIN_IN]
N = 2
a = 0.

model = nn.Conv2D(1,3)
trainer = nn.Trainer2D(model, TRAIN_IN, TRAIN_OUT)

ts = time.time()
errors = trainer.train(N, a)
duration = time.time() - ts
print(f"Training of model complete in {duration / N} seconds per epoch.")

plt.title("Training Error")
plt.plot(range(len(errors)), errors)
plt.show()