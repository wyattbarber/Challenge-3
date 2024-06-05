import neuralnet as nn
# Import other libraries
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)
import pickle
import time

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
N = 5

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
ts = time.time()
trainer = nn.Trainer(model, TRAIN_IN, TRAIN_OUT)
errors = trainer.train(N, 0.0001)
print(f"Training complete in {time.time() - ts} seconds")

# print("Training Statically Defined Model")
# ts = time.time()
# trainer = nn.Trainer(nn.StaticModel, TRAIN_IN, TRAIN_OUT)
# errors = trainer.train(N, 0.0001)
# print(f"Training complete in {time.time() - ts} seconds")

print("Building Adam Optimized Model")
model = nn.Sequence.new(
    nn.ReLU.new(len(TRAIN_IN[0]), 500),
    nn.ReLU.new(500, 300),
    nn.ReLU.new(300, 300),
    nn.ReLU.new(300, 100),
    nn.ReLU.new(100, 50),
    nn.SoftMax.new(50, 10),
)
optimizer = nn.AdamOptimizer.new(0.9, 0.999)
model.apply_optimizer(optimizer)

print("Training Optimized Model")
ts = time.time()
trainer = nn.Trainer(model, TRAIN_IN, TRAIN_OUT)
errors_adam = trainer.train(N, 0.0001)
print(f"Training complete in {time.time() - ts} seconds")

plt.title("Training Error")
plt.plot(
    range(len(errors)), [round(e, 4) for e in errors], 'r',
    range(len(errors_adam)), [round(e, 4) for e in errors_adam], 'b'
    )
plt.show()