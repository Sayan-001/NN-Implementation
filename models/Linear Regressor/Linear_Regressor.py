import sys

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

class LinearRegressor:
    """
    A simple Linear Regressor class for regression tasks. The Linear Regressor has a single layer with linear activation.
    It has the same properties as that of a polynomial regression model.

    Attributes:
    - w: numpy array, the weights of the Linear Regressor
    - b: float, the bias of the Linear Regressor

    Methods:
    - __init__(self, input_dim): Initializes the Linear Regressor with random weights and bias
    - forward(self, X): Calculates the predicted value for the given input
    - backward(self, X, y, y_hat, lr): Calculates the gradient and updates the weights and bias
    - train(self, X_train, y_train, X_test, y_test, epochs=25, lr=0.0001): Trains the neural network
    """

    def __init__(self, no_of_features):
        """
        Initializes the neural network with random weights and bias.

        Parameters:
        - input_dim: int, the dimension of the input data
        """
        
        self.w = np.random.random((no_of_features))
        self.b = np.random.random()
        
    def __repr__(self):
        return f"NN(input_dim={self.w.shape[0]})"
    
    def parameters(self):
        """
        Returns the weights and bias of the neural network.

        Returns:
        - tuple, the weights and bias
        """
        return {"weights": self.w, "bias": self.b}

    def forward(self, X):
        """
        Calculates the predicted value for the given input.

        Parameters:
        - X: numpy array, the input data

        Returns:
        - numpy array, the predicted value
        """
        
        return np.dot(X, self.w) + self.b

    def backward(self, X, y, y_hat, lr):
        """
        Calculates the gradient and updates the weights and bias.

        Parameters:
        - X: numpy array, the input data
        - y: numpy array, the target values
        - y_hat: numpy array, the predicted values
        - lr: float, the learning rate

        Returns:
        - None
        """
        
        w_grad = np.dot(X.T, y_hat - y)
        b_grad = np.mean(y_hat - y)
        self.w -= lr * w_grad
        self.b -= lr * b_grad

    def train(self, X_train, y_train, X_test, y_test, epochs=25, lr=0.0001, log=True):
        """
        Trains the neural network.

        Parameters:
        - X_train: numpy array, the training input data
        - y_train: numpy array, the training target values
        - X_test: numpy array, the test input data
        - y_test: numpy array, the test target values
        - epochs: int, the number of training epochs (default: 25)
        - lr: float, the learning rate (default: 0.0001)
        - log: bool, whether to log the training progress (default: True)

        Returns:
        - dict, a dictionary containing the training and validation loss history
        """
        
        history = {'loss': [], 'val_loss': []}

        for epoch in range(1, epochs+1):
            y_hat = self.forward(X_train)
            self.backward(X_train, y_train, y_hat, lr)

            train_pred = self.forward(X_train)
            val_pred = self.forward(X_test)
            loss = np.sqrt(np.mean((train_pred - y_train) ** 2))
            val_loss = np.sqrt(np.mean((val_pred - y_test) ** 2))

            history['loss'].append(loss)
            history['val_loss'].append(val_loss)
            
            if log:
                print(f"Epoch {epoch}: \t train_loss: {loss: .4f} \t val_loss: {val_loss: .4f}")

        return history
    
    def predict(self, X):
        """
        Predicts the target values for the given input data.

        Parameters:
        - X: numpy array, the input data

        Returns:
        - numpy array, the predicted target values
        """
        
        try:
            return self.forward(X)
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def plot_loss(self, history):
        """
        Plots the training and validation loss history.

        Parameters:
        - history: dict, a dictionary containing the training and validation loss history

        Returns:
        - None
        """
        
        plt.figure(figsize=(16, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='train_loss', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history['val_loss'], label='val_loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    
    X, y = make_regression(n_samples=5000, n_features=200, n_informative=100, noise=0.3)
    
    scaler = MinMaxScaler()
    scaler.fit(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    model = LinearRegressor(X.shape[1])
    history = model.train(X_train, y_train, X_test, y_test, epochs=10, lr=0.0001, log=True)