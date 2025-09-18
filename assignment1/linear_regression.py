import numpy as np

# Compute Mean Squared Error (MSE) between predicted and actual value.
def mean_squared_error(m, y, y_pred):
  return 1/m * np.sum((y - y_pred)**2)     # m = total number of samples.


class LinearRegression:

  def __init__(self, learning_rate, epochs):
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.weight, self.bias = None, None

  # Estimates parameters for the model using gradient descent.
  # Args:
  # X (array<m, n>): matrix of floats with m samples and n features.
  # y (array<m>): target values - vector of floats
  def fit(self, x, y):

    m, n = x.shape

    self.weight = np.zeros(n)
    self.bias = 0

    for epoch in range(self.epochs):
      y_pred = np.dot(x, self.weight) + self.bias # Predict result using linear regression.
      deviation = y - y_pred # The difference between the actual value and the prediction.

      # Find gradients by partial derive the loss function with respect to model parameters.
      # With respect to w: 1/m * Σ [(y - (w*x + b))^2] -> 1/m * Σ [2 * (y - (w*x + b)) * (-x)] -> -2/m * Σ [x * (y - (w*x + b))]
      # With respect to b: 1/m * Σ [(y - (w*x + b))^2] -> 1/m * Σ [2 * (y - (w*x + b)) * (-1)] -> -2/m * Σ [y - (w*x + b)]
      gradient_w = -2/m * np.dot(x.T, deviation) # the sum is performed inside the dot product.
      gradient_b = -2/m * np.sum(deviation)

      # Update parameters using gradient decent.
      self.weight = self.weight - self.learning_rate *  gradient_w
      self.bias = self.bias - self.learning_rate * gradient_b

      # Print loss
      if epoch % 2000 == 0:
        loss = mean_squared_error(m, y, y_pred)
        print("epoch:{}, loss:{}".format(epoch, loss))

  # Generates predictions.
  def predict(self, x):
    return np.dot(x, self.weight) + self.bias