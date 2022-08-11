import numpy as np
from abc import ABC, abstractmethod

epsilon = 10 ** -3


class Layer(ABC):

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad_input: np.ndarray) -> np.ndarray:
        pass


class ParamLayer(Layer, ABC):

    @abstractmethod
    def apply_gradients(self, learning_rate: float) -> None:
        pass


class LinearLayer(ParamLayer):

    def __init__(self, input_dim, output_dim):
        self.weights = np.random.normal(
            0.0, 0.1, size=input_dim*output_dim).reshape(input_dim, output_dim)
        self.biases = np.zeros((1, output_dim))

        self.x = np.zeros(0)

        self.grad_weights = np.zeros(0)
        self.grad_biases = np.zeros(0)

    def forward(self, x):
        self.x = x
        return (self.x @ self.weights) + self.biases

    def backward(self, grad_input):
        self.grad_weights = self.x.T @ grad_input
        self.grad_biases = np.sum(grad_input, axis=0, keepdims=True)
        return (grad_input @ self.weights.T)

    def apply_gradients(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases


class SigmoidLayer(Layer):

    def __init__(self):
        self.activation = np.zeros(0)

    def forward(self, x):
        self.activation = 1 / (1 + np.exp(-x))
        return self.activation

    def backward(self, grad_input):
        return (self.activation * (1 - self.activation)) * grad_input


class RectifiedLinearUnitsLayer(Layer):

    def __init__(self):
        self.activation = np.zeros(0)

    def forward(self, x):
        self.activation = np.max(0, x)
        return self.activation

    def backward(self, grad_input):
        return grad_input if self.activation > 0 else 0


class SquaredErrorLayer(Layer):

    def __init__(self, y):
        self.y = y
        self.x = np.zeros(0)

    def forward(self, x):
        self.x = x
        return np.square(x - self.y)

    def backward(self, grad_input):
        return 2.0 * (self.x - self.y) * grad_input


class SoftmaxLayer(Layer):

    def __init__(self, y):
        self.y = y
        self.x = np.zeros(0)

    def forward(self, x):
        self.x = np.exp(x) / np.sum(np.exp(x) + 0.0001, axis=1, keepdims=True)
        return self.x

    def backward(self, grad_input):
        softmax = self.x.reshape(-1, 1)
        return grad_input * (np.diagflat(softmax) - np.dot(softmax, softmax.T))


class CrossEntropyLoss(Layer):

    def __init__(self, y):
        self.y = np.clip(y, epsilon, 1.0 - epsilon)
        self.p = np.zeros(0)

    def forward(self, x):
        self.p = np.clip(x, epsilon, 1.0 - epsilon)
        return -(self.y * np.log(self.p) + (1-self.y) * np.log(1 - self.p))

    def backward(self, grad_input):
        layer_error = (self.p - self.y) / (self.p - self.p ** 2)
        return layer_error * grad_input


class SoftMaxCrossEntropyLoss(Layer):

    def __init__(self, y):
        self.y = np.clip(y, epsilon, 1.0 - epsilon)
        self.x = np.zeros(0)

    def forward(self, x):
        self.x = np.clip(x, epsilon, 1.0 - epsilon)
        return -(np.sum(self.y * np.log(x), axis=0))

    def backward(self, grad_input):
        layer_error = self.x - self.y
        return layer_error * grad_input
