# IMPORT LIBRARIES
import numpy as np

class Perceptron:
  def __init__(self, eta, epochs):
    self.eta = eta # LEARNING RATE
    self.epochs = epochs
    self.weights = np.random.randn(3) * 1e-4 # RANDOMLY INITIALIZING SMALL WEIGHTS
    print(f"Initial weights before training: \n{self.weights}")

  def activationFunction(self, inputs, weights):
    z = np.dot(inputs, weights) # z = w0(-1) + w1x1 + w2x2
    return np.where(z > 0, 1, 0) # CONDITION, IF TRUE, ELSE

  def fit(self, x, y):
    self.x = x
    self.y = y

    x_with_bias = np.c_[self.x, -np.ones((len(self.x), 1))]
    print(f"x with bias: \n{x_with_bias}")

    for epoch in range(1, self.epochs + 1):
      print("--" * 10)
      print(f"for epoch: {epoch}")
      print("--" * 10)

      y_cap = self.activationFunction(x_with_bias, self.weights) # FORWARD PROPOGATION
      print(f"Predicted values after forward pass: \n{y_cap}")

      self.error = self.y - y_cap # CALCULATING ERROR
      print(f"Error: \n{self.error}")

      self.weights = self.weights + self.eta * np.dot(x_with_bias.T, self.error)
      print(f"Updated weights after epoch:\n{epoch}/{self.epochs}:\n{self.weights}") # UPDATING WEIGHTS ON THE BASIS OF ERRORS
      print("##" * 10)

  def predict(self, x):
    x_with_bias = np.c_[x, -np.ones((len(x), 1))]
    return self.activationFunction(x_with_bias, self.weights)

  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f'Total Loss:\n{total_loss}')