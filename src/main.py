import matplotlib.pyplot as plt
import numpy as np
import keras
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
X_train = np.reshape(np.asarray(X, dtype=np.uint8), (60000, 28, 28, 1))
Y_train = np.reshape(np.asarray(Y, dtype=np.uint8), (60000,))
# Normalize data for better results
X_train = X_train.astype('float32')/255
# Convert labels to categoricals (0 -> [1 0 0 0 0 0 0 0 0 0])
Y_train = keras.utils.to_categorical(Y_train, num_classes=10)

# Reshape test data
X, Y = mnist_data.load_testing()
X_test = np.reshape(np.asarray(X, dtype=np.uint8), (10000, 28, 28, 1))
Y_test = np.reshape(np.asarray(Y, dtype=np.uint8), (10000,))
# Normalize data for better results
X_test = X_test.astype('float32')/255
# Convert labels to categoricals (5 -> [0 0 0 0 0 1 0 0 0 0])
Y_test = keras.utils.to_categorical(Y_test, num_classes=10)

# Create list of parameters to test
models_params = [
  {
    'name': 'padrao-4-6-500',
    'parameters': {
      'first_layer': {'filters': 4, 'kernel_size': 5},
      'second_layer': {'filters': 6, 'kernel_size': 5},
      'mlp_neurons': 500
    }
  },
  {
    'name': 'mudar_filtros-6-4-500',
    'parameters': {
      'first_layer': {'filters': 6, 'kernel_size': 5},
      'second_layer': {'filters': 4, 'kernel_size': 5},
      'mlp_neurons': 500
    }
  },
  {
    'name': 'mudar_mlp-4-6-700',
    'parameters': {
      'first_layer': {'filters': 4, 'kernel_size': 5},
      'second_layer': {'filters': 6, 'kernel_size': 5},
      'mlp_neurons': 700
    }
  },
  {
    'name': 'mudar_mlp-4-6-300',
    'parameters': {
      'first_layer': {'filters': 4, 'kernel_size': 5},
      'second_layer': {'filters': 6, 'kernel_size': 5},
      'mlp_neurons': 300
    }
  },
  {
    'name': 'mudar_kernel-4-6-500',
    'parameters': {
      'first_layer': {'filters': 4, 'kernel_size': 7},
      'second_layer': {'filters': 6, 'kernel_size': 7},
      'mlp_neurons': 500
    }
  },
]

# Create and test each model, saving them
for model_params in models_params:
  model = MnistModel(model_params['name'])
  model.generate_model(**model_params['parameters'])
  model.save_model()
  history = model.train(X_train, Y_train)
  print(history.history)
  metrics = model.calculate_metrics(X_test, Y_test)
  print(model.get_metrics_names())
  print(metrics)
