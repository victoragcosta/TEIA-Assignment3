import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout

class MnistModel(object):
  def __init__(self):
    # Define a model
    model = Sequential()

    # Add a convolutional layer
    model.add(Conv2D(4, 4, input_shape=(28,28,1), data_format="channels_last"))
    # a pooling layer
    model.add(MaxPooling2D(pool_size=(2,2)))
    # and a dropout layer
    model.add(Dropout(0.25))

    # Add a convolutional layer
    model.add(Conv2D(4, 4))
    # a pooling layer
    model.add(MaxPooling2D(pool_size=(2,2)))
    # and a dropout layer
    model.add(Dropout(0.25))

    # Make matrix become flat
    model.add(Flatten())

    # Add a Multilayer Perceptron for classification
    for hl in [200]:
      model.add(Dense(hl))

    # Add the output layer
    model.add(Dense(10))

    # Add a softmax activation layer
    model.add(Activation("softmax"))

    # Finish creating the model
    model.compile(
      loss=keras.losses.categorical_crossentropy,
      optimizer=keras.optimizers.Adadelta(),
      metrics=['accuracy']
    )
    self.model = model

  def train(self, X_train, Y_train, X_valid, Y_valid, batch_size=64, epochs=100):
    out = self.model.fit(
      X_train, Y_train,
      batch_size=batch_size,
      epochs=epochs,
      verbose=1,
      validation_data=(X_valid, Y_valid)
    )
    return out

  def predict(self, X_test):
    return self.model.predict(X_test, batch_size=50)

  def calculate_accuracy(self, X_test, Y_test):
    out = self.predict(X_test)

  @staticmethod
  def convert_label(Y :np.ndarray):
    # Create matrix with number of examples lines and 10 columns (10 possible digits 0-9)
    new = np.zeros((*Y.shape, 10), dtype=np.uint8)
    # For each line select the label column and mark as 1
    new[np.arange(Y.shape[0]), Y] = 1
    return new
