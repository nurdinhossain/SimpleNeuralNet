import numpy as np
import copy

def sigmoid(node, der=False):
  sig = 1/(1 + np.exp(-node))
  if der:
    return sig * (1 - sig)
  else:
    return sig 

def relu(x, der=False):
  if der:
    new_arr = copy.deepcopy(x)
    new_arr[new_arr<=0] = 0
    new_arr[new_arr>0] = 1
    return new_arr
  else:
    return np.maximum(x, 0)

def leaky_relu(x, der=False):
  if der:
    return relu(x, der=True)
  else:
    return np.maximum(x, 0.1)

def tanh(x, der=False):
  if der:
    return (2 * np.exp(x)) / (np.exp(2 * x) + 1)
  else:
    return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)

def linear(x, der=False):
  if der:
    return np.ones(x.shape)
  else:
    return x

def softmax(x, der=False):
    if der:
      normal = softmax(x)
      soft_der = np.zeros((x.shape[0], x.shape[0]))
      for i in range(len(soft_der)):
        for j in range(len(soft_der[i])):
          if i == j:
            soft_der[i][j] = normal[i] * (1 - normal[i])
          else:
            soft_der[i][j] = -normal[i] * normal[j]

      return np.array(soft_der)
    else:
      exps = np.exp(x - np.max(x))
      return exps / np.sum(exps)
