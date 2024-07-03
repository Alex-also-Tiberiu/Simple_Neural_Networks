import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class Neuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(self, inputs):
    total = np.dot(self.weights, inputs) + self.bias
    return sigmoid(total)

class Simple_Network:
    '''
    A neural network with:
    - 2 inputs
    - 1 hidden layer with 2 neurons (h1, h2)
    - 1 output layer with 1 neuron (o1)
    Each neuron has the same weights and bias:
    - w = [0,1]
    - b = 0
    '''
    def __init__(self):
        weights = np.array([0,1])
        bias = 0

        # Declare neurons of the network
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        out_1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_1

network = Simple_Network()
x = np.array([2,3])
print(network.feedforward(x))