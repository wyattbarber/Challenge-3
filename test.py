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
# TRAIN_OUT = []

# print("Preprocessing dataset")
# for i in range(images.shape[1]):
#     TRAIN_IN.append(images[:,i] / 255)
#     target = [0] * 10
#     target[targets[i]] = 1
#     TRAIN_OUT.append(target)

# print("Saving pickled dataset")
# pickle.dump((TRAIN_IN, TRAIN_OUT), open('data/mnist_preprocessed.pickle', 'wb'))

print("Loading pickled data")
TRAIN_IN, TRAIN_OUT = pickle.load(open('data/mnist_preprocessed.pickle', 'rb'))

print("Building Model")
model = nn.Sequence.new(
    nn.ReLU.new(len(TRAIN_IN[0]), 500),
    nn.ReLU.new(500, 300),
    nn.ReLU.new(300, 300),
    nn.ReLU.new(300, 100),
    nn.ReLU.new(100, 50),
    nn.SoftMax.new(50, 10)
)

print("Training")
trainer = nn.Trainer(model, TRAIN_IN, TRAIN_OUT)
errors = trainer.train(3, 0.001)

plt.title("Training Error")
plt.plot(range(len(errors)), errors)
plt.show()