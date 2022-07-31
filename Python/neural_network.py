from abc import abstractmethod
import numpy as np


class NeuralNetwork():
    def __init__(self, layers):
        pass

    def predict(self, X, y):
        pass

    def fit(self, X_train, y_train, epochs=1000, lr=0.01):
        pass

    def backward_propagation(self, output):
        pass

    def update_params(self, lr):
        pass

    def  evaluate(self, X_test, y_test):
        pass


class Layer():
    pass


class LinearLayer(Layer):
    pass


class Sigmoid(Layer):
    pass


class ReLU(Layer):
    pass


class BatchNorm(Layer):
    pass


class Softmax(Layer):
    pass


class Optimizer():
    pass


class ElasticReg():
    pass


class GradientDescent(Optimizer):
    pass


class ADAM(Optimizer):
    pass


if __name__ == '__main__':
    neural_network = NeuralNetwork()
    print('End')