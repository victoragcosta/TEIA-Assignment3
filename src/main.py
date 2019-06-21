import argparse

parser = argparse.ArgumentParser(
  description='Train and visualize a Convolutional Neural Network for the MNIST database.'
)
parser.add_argument(
  '-t', '--train', dest='train', action='store_true',
  help='WWhen present, plots the activation of a example on the convolutional layers.'
)
parser.add_argument(
  '-g', '--render-graphs', dest='graphs', action='store_true',
  help='When present, creates validation evolution graphs and prints the test results.'
)
parser.add_argument(
  '-f', '--render-filters', dest='filters', action='store_true',
  help='When present, plots the filters trained on the convolutional layers.'
)
parser.add_argument(
  '-a', '--render-activations', dest='activation', action='store_true',
  help='When present, plots the activation of a example on the convolutional layers.'
)
parser.add_argument(
  '-w', '--render-wrong',dest='wrong', action='store_true',
  help='When present, plots a wrong guess with the expected value and the predicted value.'
)
parser.add_argument(
  '-s', '--show-plots', dest='show', action='store_true',
  help='When present, opens many windows with the requested renders.'
)
parser.add_argument(
  '-o', '--output-plots', dest='save_img', action='store_true',
  help='When present, saves the requested renders to images on the img folder.'
)

args = parser.parse_args()

# Imports libs after parsing for better help performance
import matplotlib.pyplot as plt
import numpy as np
from cnn import *
from functions import *

train = args.train
graphs = args.graphs
filters = args.filters
activation = args.activation
wrong = args.wrong
show = args.show
save_img = args.save_img

# Extract training data and testing data
(X_train, Y_train), (X_test, Y_test) = extract_data('data')

# Get all models to test
models_params = get_test_models()

# If I am to calculate the validation metrics evolution graphs
if graphs:
  # Decide to train or use already trained models
  if train:
    # Trains and return metrics data
    data = train_models(models_params, X_train, Y_train, X_test, Y_test)
  else:
    # Loads and return metrics data
    data = load_models(models_params, X_test, Y_test)
  # Print test results and plot all data
  present_metrics(data)

# Load the best model for the visualizations
model = MnistModel('padrao-4-6-500')
model.load_model()

if filters:
  # Plot all the filters of the model
  visualize_filters(model)

if activation:
  # Plot the outputs of the convolutional layers
  visualize_activations(model, X_test[0:1,:,:,:])

if wrong:
  # Plot some wrong guesses with the expected results and predicted values
  visualize_wrong(model, X_test, Y_test)

if (graphs or filters or activation or wrong) and save_img:
  # Find all figures plotted and save them
  save_all_figs()

if (graphs or filters or activation or wrong) and show:
  # Show every single plot
  plt.show()
