from abc import abstractmethod
from typing import Union
from tqdm import tqdm

import numpy as np
import pandas as pd

from layers import CrossEntropyLoss, Layer, LinearLayer, ParamLayer, SigmoidLayer, SoftMaxCrossEntropyLoss, SoftmaxLayer


class NeuralNetwork():
    def __init__(self, layers):
        self.layers = layers
        self.loss_layer: Union[Layer, None] = None

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def fit(self, X_train, y_train, x_test, y_test, epochs=1000, learning_rate=0.01):
        metrics = np.zeros((epochs, 2))  # loss, accuracy
        for epoch in tqdm(range(epochs)):
            y_pred = self.predict(X_train)
            self.loss_layer = SoftMaxCrossEntropyLoss(y_train)
            self.loss_layer.forward(y_pred)
            self.backward_propagation(y_pred)
            self.update_params(learning_rate)

            loss, accuracy = self.evaluate(x_test, y_test)
            metrics[epoch, :] = loss, accuracy

        return metrics

    def backward_propagation(self, output):
        grad_input = self.loss_layer.backward(np.ones_like(output))
        for layer in reversed(self.layers):
            grad_input = layer.backward(grad_input)

    def update_params(self, learning_rate):
        for layer_index in range(len(self.layers)):
            layer = self.layers[layer_index]
            if isinstance(layer, ParamLayer):
                layer.apply_gradients(learning_rate)

    def evaluate(self, X_test, y_test):
        self.loss_layer = CrossEntropyLoss(y_test)
        y_pred = self.predict(X_test)

        y_pred_labels = np.argmax(y_pred, axis=1)
        y_labels = np.argmax(y_test, axis=1)

        loss = self.loss_layer.forward(y_pred)
        accuracy = np.sum(y_labels == y_pred_labels) / X_test.shape[0]

        return float(np.mean(loss)), accuracy


def load_data():
    train = pd.read_csv('dataset/mnist_train.csv')
    test = pd.read_csv('dataset/mnist_test.csv')

    x_train = train.iloc[:, 1:].to_numpy()
    y_train = train.iloc[:, 0].to_numpy()
    x_test = test.iloc[:, 1:].to_numpy()
    y_test = test.iloc[:, 0].to_numpy()

    return x_train, y_train, x_test, y_test


def scale(x_train, x_test):
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, x_test


def one_hot(y_train, y_test):
    y_train = np.identity(10)[y_train.reshape(-1)]
    y_test = np.identity(10)[y_test.reshape(-1)]

    return y_train, y_test


def create_small_dataset(x_train, y_train, x_test, y_test):
    num_train = 1000
    num_dev = 200

    small_x_train = x_train[:num_train, :]
    small_y_train = y_train[:num_train, :]
    small_x_test = x_test[:num_dev, :]
    small_y_test = y_test[:num_dev, :]

    return small_x_train, small_y_train, small_x_test, small_y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    y_train, y_test = scale(x_train, x_test)
    y_train, y_test = one_hot(y_train, y_test)

    small_x_train, small_y_train, small_x_test, small_y_test = create_small_dataset(
        x_train, y_train, x_test, y_test)

    neural_network = NeuralNetwork(layers=[
        LinearLayer(small_x_train.shape[1], 30),
        SigmoidLayer(),
        LinearLayer(30, 10),
        SigmoidLayer(),
        LinearLayer(10, small_y_train.shape[1]),
        SigmoidLayer(),
    ])

    metrics = neural_network.fit(small_x_train, small_y_train,
                       small_x_test, small_y_test, epochs=1000)

    print(metrics)
    
    example = 300
    output = neural_network.predict(x_test[example, :])
    print(y_test[example,:])
    print(output)
    print(np.argmax(y_train))
    print(np.argmax(output))
