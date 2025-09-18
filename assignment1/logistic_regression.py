import numpy as np

# Converts any real number into a value between 0 and 1, making it ideal for representing probabilities.
def sigmoid(z):
  return 1 / (1 + np.exp(-z))

# Compute cross-entropy loss. Measures how far the predicted probabilities are from the true labels.
# There is different forms of cross-entropy loss depending on whether the classification problem is binary of multicast.
# Binary cross-entropy is used for when the target has only to classes, 0 and 1. While categorical cross-entropy (Multiclass)
# is used when the target has more than two classes, for example when the labels are one-hot encoded.
# The formula used is for binary cross-entropy loss.
def cross_entropy_loss(m, y, y_pred):
  return -1/m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


class LogisticRegression:
  def __init__(self, learning_rate, epochs):
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.weight, self.bias = None, None

  def fit(self, x, y):
     m, n = x.shape

     self.weight = np.zeros(n) * 0.01
     self.bias = 0

     for epoch in range(self.epochs):
       z = np.dot(x, self.weight) + self.bias # Linear regression.
       y_pred = sigmoid(z) # Combine linear regression with sigmoid to get probability between 0 and 1.

       deviation = y_pred - y

       # Compute gradients (partial derivative of the loss function )
       gradient_w = 1/m * np.dot(x.T, deviation)
       gradient_b = 1/m * np.sum(deviation)

       self.weight = self.weight - self.learning_rate * gradient_w
       self.bias = self.bias - self.learning_rate * gradient_b

       if epoch % 100 == 0:
         loss = cross_entropy_loss(m, y, y_pred)
         print("epoch:{}, loss:{}".format(epoch, loss))

  def predict_probability(self, x):
    z = np.dot(x, self.weight) + self.bias
    return sigmoid(z)

  def predict(self, x, threshold):
    y_pred = self.predict_probability(x)
    return [1 if y >= threshold else 0 for y in y_pred]