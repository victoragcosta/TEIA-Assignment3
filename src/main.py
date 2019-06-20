import matplotlib.pyplot as plt
import numpy as np
from cnn import *
from functions import *

calculate_graphs = False

(X_train, Y_train), (X_test, Y_test) = extract_data('data')
if calculate_graphs:
  models_params = get_test_models()

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

## Visualizing the trained filters ##
# from src.cnn import *
# from src.functions import *

best_model_name = 'padrao-4-6-500'
model = MnistModel(best_model_name)
model.load_model()

# Load filters for showing
layers = model.get_filters()

# Show all neurons filters of the first layer in a single plot
layer = 'first_conv'
fig = plt.figure(layer+'-filters-all_neurons')
axes = fig.subplots(2,2)
axes = list(axes.flatten())
filt = layers[layer][0]
ax_num = 0
for i in range(filt.shape[2]):
  for o in range(filt.shape[3]):
    axes[ax_num].matshow(filt[:,:,i:i+1,o:o+1].squeeze(), cmap='gray')
    axes[ax_num].set_title('{}-{}-{}'.format(layer, i, o))
    remove_marks(axes[ax_num])
    ax_num += 1

# Show all 4 channels of each neuron of the second layer in a single plot
layer = 'second_conv'
filt = layers[layer][0]
for o in range(filt.shape[3]):
  fig = plt.figure(layer+'-filters-neuron_'+str(o))
  axes = fig.subplots(2,2)
  axes = list(axes.flatten())
  ax_num = 0
  for i in range(filt.shape[2]):
    axes[ax_num].matshow(filt[:,:,i:i+1,o:o+1].squeeze(), cmap='gray')
    axes[ax_num].set_title('{}-{}-{}'.format(layer, i, o))
    remove_marks(axes[ax_num])
    ax_num += 1

# Save the results
for name in plt.get_figlabels():
  fig = plt.figure(name)
  fig.savefig('img/{}.png'.format(name))

# Show the results!
# plt.show()

# Present the activations for a example
X = X_test[1:2,:,:,:]
layer = 'first_conv'
activation = model.get_activation(X, layer_name=layer)
fig = plt.figure(layer+'-neurons_activation')
axes = fig.subplots(2,2)
axes = list(axes.flatten())
for neuron in range(activation.shape[3]):
  axes[neuron].matshow(activation[0:1,:,:,neuron:neuron+1].squeeze(), cmap='gray')
  axes[neuron].set_title('{}-neuron_{}'.format(layer, neuron))
  remove_marks(axes[neuron])

layer = 'second_conv'
activation = model.get_activation(X, layer_name='second_conv')
fig = plt.figure(layer+'-neurons_activation')
axes = fig.subplots(2,3)
axes = list(axes.flatten())
for neuron in range(activation.shape[3]):
  axes[neuron].matshow(activation[0:1,:,:,neuron:neuron+1].squeeze(), cmap='gray')
  axes[neuron].set_title('{}-neuron_{}'.format(layer, neuron), fontsize=10.5)
  remove_marks(axes[neuron])

# Save the activations to figures
for name in plt.get_figlabels():
  fig = plt.figure(name)
  fig.savefig('img/{}.png'.format(name))

# Show the activations
# plt.show()

# TODO: Apresentar imagens mal classificadas (melhor modelo)
