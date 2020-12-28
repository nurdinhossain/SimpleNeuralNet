import numpy as np

def mse(network, data, label):
  # propagate forward
  network.forward_prop(data)

  # guess
  flattened_label = list(label.flatten())
  flattened_output = list(network.layers[-1].flatten())
  guess = flattened_label.index(max(flattened_label)) == flattened_output.index(max(flattened_output))

  # calculate error derivative
  error = network.layers[-1] - label.reshape(network.layers[-1].shape)
  return error, guess

def sparse_crossentropy(network, data, label):
  # set up layer label
  layer_label = np.zeros(network.layers[-1].shape)
  layer_label[label] = 1

  # propagate forward
  network.forward_prop(data)
  
  # guess
  flattened = list(network.layers[-1].flatten())
  guess = flattened.index(max(flattened)) == label

  # calculate error derivative
  error = network.layers[-1] - layer_label
  return error, guess

def crossentropy(network, data, label):
  # propagate forward
  network.forward_prop(data)

  # guess
  flattened_label = list(label.flatten())
  flattened_output = list(network.layers[-1].flatten())
  guess = flattened_label.index(max(flattened_label)) == flattened_output.index(max(flattened_output))

  # calculate error derivative
  error = network.layers[-1] - label.reshape(network.layers[-1].shape)
  return error, guess

def binary_crossentropy(network, data, label):
  # set up layer label
  label = np.array([label]).reshape(network.layers[-1].shape)

  # propagate forward
  network.forward_prop(data)
  
  # calculate error derivative
  error = (network.layers[-1] - label) / (network.layers[-1] * (1 - network.layers[-1]))
  return error
