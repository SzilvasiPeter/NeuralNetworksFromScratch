from abc import abstractmethod
from typing import Union
import numpy as np
import tqdm

from layers import Layer, ParamLayer, SoftMaxCrossEntropyLoss, SoftmaxLayer


class NeuralNetwork():
    def __init__(self, layers):
        self.layers = layers
        self.loss_layer: Union[Layer, None] = None

    def predict(self, X):
        for layer in self.layers:
            output = layer.forward(X)
        return output

    def fit(self, X_train, y_train, x_test, y_test, epochs=1000, learning_rate=0.01):
        metrics = np.zeros((epochs, 2))  # loss, accuracy
        self.loss_layer = SoftMaxCrossEntropyLoss(y_train)
        for epoch in tqdm(range(epochs)):
            y_pred = self.predict(X_train)
            self.loss_layer.forward(y_pred)
            self.backward_propagation(y_pred)
            self.update_params(learning_rate)

            loss, accuracy = self.evaluate_model(x_test, y_test)
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

    def  evaluate(self, X_test, y_test):
        self.loss_layer = SoftmaxLayer(y_test)
        y_pred = self.predict(X_test)

        y_pred_labels = np.argmax(y_pred, axis=1)
        y_labels = np.argmax(y_test, axis=1)

        loss = self.loss_layer.forward(y_pred)
        accuracy = np.sum(y_labels == y_pred_labels) / X_test.shape[0]
        return float(np.mean(loss)), accuracy


if __name__ == '__main__':
    pass