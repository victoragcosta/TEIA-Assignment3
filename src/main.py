import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from mnist import MNIST
import os.path
# from src.cnn import *
from cnn import *

# Extract data
mnist_data = MNIST(os.path.abspath('data'))

# Reshape training data
X, Y = mnist_data.load_training()
X_training = np.reshape(np.asarray(X, dtype=np.uint8), (60000, 28, 28, 1))
Y_training = np.reshape(np.asarray(Y, dtype=np.uint8), (60000,))
Y_training = MnistModel.convert_label(Y_training)

# Reshape test data
X, Y = mnist_data.load_testing()
X_test = np.reshape(np.asarray(X, dtype=np.uint8), (10000, 28, 28, 1))
Y_test = np.reshape(np.asarray(Y, dtype=np.uint8), (10000,))
Y_test = MnistModel.convert_label(Y_test)

print(Y_training[0:1])
plt.matshow(X_training[0:1,:,:].squeeze())
plt.show()

model = MnistModel()
model.train(X_training, Y_training, X_test, Y_test)
