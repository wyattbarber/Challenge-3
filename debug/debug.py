from . import neuralnet as nn
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import sys

np.random.seed(123)
TRAIN_IN, TRAIN_OUT = pickle.load(open(f'{sys.argv[1]}/../data/mnist_preprocessed.pickle', 'rb'))
N = 2
a = 0.0001
AdamArgs = (0.9, 0.999)
layers = [nn.AutoEncoder(784, 50, *AdamArgs)]
model = nn.Sequence(layers)
ts = time.time()
errors = nn.train(model, TRAIN_IN, TRAIN_IN, N, a)
# errors = nn.fixed_encoder(TRAIN_IN, N, a)
duration = time.time() - ts
print(f"Training of coupled autoencoder complete in {duration / N} seconds per epoch.")