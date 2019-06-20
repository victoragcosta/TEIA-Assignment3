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
    'name': 'mudar_kernel7-4-6-500',
    'parameters': {
      'first_layer': {'filters': 4, 'kernel_size': 7},
      'second_layer': {'filters': 6, 'kernel_size': 7},
      'mlp_neurons': 500
    }
  },
  {
    'name': 'mudar_kernel3-4-6-500',
    'parameters': {
      'first_layer': {'filters': 4, 'kernel_size': 3},
      'second_layer': {'filters': 6, 'kernel_size': 3},
      'mlp_neurons': 500
    }
  },
]

# fig, (loss, accuracy, precision) = plt.subplots(3,1)
# fig, (loss, accuracy, precision) = plt.subplots(1,3)
loss = plt.figure("Loss").gca()
accuracy = plt.figure("Accuracy").gca()
precision = plt.figure("Precision").gca()

# Create and test each model, saving them
for model_params in models_params:
  name = model_params['name']
  params = model_params['parameters']

  # Create the Model for the mnist
  model = MnistModel(name)
  try:
    # Try to load
    model.load_model()
    print("Loaded model {}. Skipping training.".format(name))
  except:
    print("Couldn't load model {}. Training a new one.".format(name))
    # If I can't load, I create a new one
    model.generate_model(**params)
    model.train(X_train, Y_train, epochs=20)
    model.save_model()

  # Load metrics
  history = model.history.copy()
  metrics = model.calculate_metrics(X_test, Y_test)
  metrics_names = model.get_metrics_names()

  # Extend metrics to all epochs
  for metric, values in history.items():
    for i in range(len(values), 20):
      values.append(values[len(values)-1])

  # Plot metrics
  loss.plot(history['val_loss'], label=name)
  accuracy.plot(history['val_acc'], label=name)
  precision.plot(history['val_precision'], label=name)

  # Print test results
  for name, metric in zip(metrics_names,metrics):
    print("{}: {}".format(name, metric))

# Prettify the graphs
loss.set_xlabel("Epochs")
loss.set_ylabel("Loss (Categorical Crossentropy)")
loss.set_title("Loss Evolution")
loss.set_ylim(0, 0.15)
loss.set_xticks(list(range(20)))
loss.legend()

accuracy.set_xlabel("Epochs")
accuracy.set_ylabel("Accuracy")
accuracy.set_title("Accuracy Evolution")
accuracy.set_ylim(0.95, 1)
accuracy.set_xticks(list(range(20)))
accuracy.legend()

precision.set_xlabel("Epochs")
precision.set_ylabel("Precision")
precision.set_title("Precision Evolution")
precision.set_ylim(0.95, 1)
precision.set_xticks(list(range(20)))
precision.legend()

# Save the graphs
for label in plt.get_figlabels():
  fig = plt.figure(label)
  fig.savefig('img/'+label+'.png')

# Show the graphs at last!
plt.show()
