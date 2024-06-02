import neuralnet as nn
# Import other libraries
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)
import pickle

# # Load training data
# print("Loading dataset")
# images = np.genfromtxt('data/mnist_train.csv', delimiter=',', skip_header=1)[:,:10000]
# targets = np.genfromtxt('data/mnist_train_targets.csv', delimiter=',', skip_header=1, dtype=int)
# TRAIN_IN = []
# TRAIN_OUT = np.zeros((10, images.shape[1]))

# print("Preprocessing dataset")
# for i in range(images.shape[1]):
#     TRAIN_IN.append(images[:,i] / 255)
#     TRAIN_OUT[targets[i], i] = 1

# print("Saving pickled dataset")
# pickle.dump((TRAIN_IN, TRAIN_OUT), open('data/mnist_preprocessed.pickle', 'wb'))

print("Loading pickled data")
TRAIN_IN, TRAIN_OUT = pickle.load(open('data/mnist_preprocessed.pickle', 'rb'))

print("Training")
errors = nn.test_results(TRAIN_IN, TRAIN_OUT, 10, 0.00001)

plt.title("Training Error")
plt.plot(range(len(errors)), errors)
plt.show()