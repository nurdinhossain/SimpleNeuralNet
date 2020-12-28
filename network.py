import numpy as np
import sys
import copy

class Network:
  def __init__(self, input_size, output_size, hidden_layers: list, activations, dropout, initialization=None):
    self.input_size = input_size
    self.output_size = output_size
    self.layers = [np.zeros((input_size, 1))] + [np.zeros((size, 1)) for size in hidden_layers] + [np.zeros((output_size, 1))]
    
    if initialization == None:
      self.weights = [np.random.uniform(-1, 1, (self.layers[layer + 1].shape[0], self.layers[layer].shape[0])) for layer in range(len(self.layers) - 1)]
      self.biases = [np.random.uniform(-1, 1, layer.shape) for layer in self.layers[1:]]
    elif initialization == 'xavier':
      self.weights = [np.random.uniform(-np.sqrt(6 / (self.layers[layer].shape[0] + self.layers[layer + 1].shape[0])), np.sqrt(6 / (self.layers[layer].shape[0] + self.layers[layer + 1].shape[0])), (self.layers[layer + 1].shape[0], self.layers[layer].shape[0])) for layer in range(len(self.layers) - 1)]
      self.biases = [np.random.uniform(-1, 1, layer.shape) for layer in self.layers[1:]]
    elif initialization == 'kaiming':
      self.weights = [np.random.normal( size=(self.layers[layer + 1].shape[0], self.layers[layer].shape[0]) ) * np.sqrt(2 / self.layers[layer].shape[0]) for layer in range(len(self.layers) - 1)]
      self.biases = [np.zeros(layer.shape) for layer in self.layers[1:]]
    
    self.m_weights = [0 for weight in self.weights]
    self.v_weights = [0 for weight in self.weights]
    self.m_biases = [0 for bias in self.biases]
    self.v_biases = [0 for bias in self.biases]
    self.iterations = 0
    self.activations = activations
    self.drops = [np.random.binomial(1, 1 - dropout[layer], self.layers[layer].shape) / (1 - dropout[layer]) for layer in range(len(self.layers))]
    if len(activations) != len(self.layers) - 1:
      raise Exception("Incorrect number of activations")

  def forward_prop(self, data):
    if type(data) == list:
      raise TypeError("Data must be numpy array")
    if len(data.flatten()) != self.input_size:
      raise ValueError("Data length does not match input dimensions - {} does not match {}".format(len(data.flatten()), self.input_size))

    # reset
    for layer in range(len(self.layers)):
      self.layers[layer] -= self.layers[layer]

    # insert data
    self.layers[0] += data.reshape(self.layers[0].shape)
    self.layers[0] *= self.drops[0]

    # propogate forward
    for layer in range(len(self.layers) - 1):
      self.layers[layer + 1] += self.activations[layer](self.weights[layer] @ self.layers[layer] + self.biases[layer])
      self.layers[layer + 1] *= self.drops[layer + 1]

    return self.layers[-1]

  def back_prop(self, data, label, loss):
    deltas = []
    
    # calculate last delta
    error, guess = loss(self, data, label)
    
    if self.activations[-1] == softmax:
      last_delta = (self.activations[-1](self.weights[-1] @ self.layers[-2] + self.biases[-1], der=True) @ error) * self.drops[-1] # MAKE THIS DYNAMIC FOR ACTIVATIONS
    else:
      last_delta = (error * self.activations[-1](self.weights[-1] @ self.layers[-2] + self.biases[-1], der=True)) * self.drops[-1] # MAKE THIS DYNAMIC FOR ACTIVATIONS
    
    deltas.append(last_delta)

    # calculate every other delta
    for layer in range(len(self.layers) - 2, 0, -1): # excluding input and output layer
      delta = ((self.weights[layer].T @ deltas[-1]) * self.activations[layer - 1](self.weights[layer - 1] @ self.layers[layer - 1] + self.biases[layer - 1], der=True)) * self.drops[layer] # MAKE THIS DYNAMIC FOR ACTIVAITIONS
      deltas.append(delta)

    # calculate gradients
    weight_gradients = []
    bias_gradients = list(reversed(deltas))
    for index, delta in enumerate(bias_gradients): # first hidden layer -> output layer
      weight_gradient = delta @ self.layers[index].T
      weight_gradients.append(weight_gradient)

    return weight_gradients, bias_gradients, guess

  def train_batch(self, data, labels, loss, batch, batch_size): #helper
      correct = 0
      total_w_gradients = [np.zeros(weight_matrix.shape) for weight_matrix in self.weights]
      total_b_gradients = [np.zeros(bias_matrix.shape) for bias_matrix in self.biases]
      for piece in range(batch_size):
        weight_gradients, bias_gradients, guess = self.back_prop(data[batch * batch_size + piece], labels[batch * batch_size + piece], loss)
        correct += guess
        for i in range(len(total_w_gradients)):
          total_w_gradients[i] += weight_gradients[i]
          total_b_gradients[i] += bias_gradients[i]

      return total_w_gradients, total_b_gradients, correct

  def train(self, data, labels, loss, iterations, batch_size=1, step_size=0.05, progress=True):
    beta_one = 0.9
    beta_two = 0.999
    epsilon = 1e-8
    for i in range(iterations):
      start = time.time()
      if progress:
        sys.stdout.write("ITERATION {}:\n".format(i + 1))
      correct = 0
      for batch in range(len(data) // batch_size):
        if progress:
          sys.stdout.write("\rBATCH {}/{} - {}% COMPLETE, {}% TRAINING ACCURACY".format(batch+1, len(data) // batch_size, round((batch+1) / (len(data) // batch_size), 4)*100, round(correct / len(data), 4) * 100 ))

        # go through batch
        w_grads, b_grads, local_c = self.train_batch(data, labels, loss, batch, batch_size)
        correct += local_c

        # adam
        w_grads = [grad / batch_size for grad in w_grads]
        for index, gradient in enumerate(w_grads):
          self.m_weights[index] = beta_one * self.m_weights[index] + (1 - beta_one) * gradient
          self.v_weights[index] = beta_two * self.v_weights[index] + (1 - beta_two) * np.power(gradient, 2)
          m_hat = self.m_weights[index] / (1 - np.power(beta_one, batch + 1 + (self.iterations * len(data) // batch_size) ))
          v_hat = self.v_weights[index] / (1 - np.power(beta_two, batch + 1 + (self.iterations * len(data) // batch_size) ))
          self.weights[index] -= step_size * m_hat / (np.sqrt(v_hat) + epsilon)

        b_grads = [grad / batch_size for grad in b_grads]
        for index, gradient in enumerate(b_grads):
          self.m_biases[index] = beta_one * self.m_biases[index] + (1 - beta_one) * gradient
          self.v_biases[index] = beta_two * self.v_biases[index] + (1 - beta_two) * np.power(gradient, 2)
          m_hat = self.m_biases[index] / (1 - np.power(beta_one, batch + 1 + (self.iterations * len(data) // batch_size) )) # i + 1 bfore
          v_hat = self.v_biases[index] / (1 - np.power(beta_two, batch + 1 + (self.iterations * len(data) // batch_size) )) # i + 1 bfore
          self.biases[index] -= step_size * m_hat / (np.sqrt(v_hat) + epsilon)

      self.iterations += 1
      if progress:
        sys.stdout.write("\n{} seconds for full iteration\n".format(time.time() - start))

  def test(self, batch: list):
    outputs = []
    for ex in batch:
      outputs.append(copy.deepcopy(self.forward_prop(ex)))

    return outputs
