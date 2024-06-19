import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

class Single_Layer_NN:
    """
    A simple neural network class for regression tasks. The neural network has multiple neurons in a single layer with linear activation.
    Supports only a single hidden layer, and the output layer has a single neuron. Classes are binary.
    
    - no_of_features (int): The number of features in the input data.
    - neurons (int): The number of neurons in the hidden layer.
    - hidden_layer_activation (str): The activation function for the hidden layer.
    - output_layer_activation (str): The activation function for the output layer.
    - w1 (ndarray): The weights of the first layer.
    - w2 (ndarray): The weights of the second layer.
    - b1 (ndarray): The bias of the first layer.
    - b2 (float): The bias of the second layer.
    
    Methods:
    - __init__(self, no_of_features, hidden_layer_activation, output_layer_activation, neurons=32): Initializes the neural network with random weights and bias.
    - _rmse(self, y, y_hat): Calculates the root mean squared error (RMSE) between the predicted values and the actual values.
    - _accuracy(self, y, y_hat): Calculates the accuracy of the predicted values.
    - _linear(self, X): The linear activation function.
    - _dlinear(self, X): The derivative of the linear activation function.
    - _sigmoid(self, X): The sigmoid activation function.
    - _dsigmoid(self, X): The derivative of the sigmoid activation function.
    - _relu(self, X): The ReLU activation function.
    - _drelu(self, X): The derivative of the ReLU activation function.
    - _select_activation(self, activation): Selects the activation function based on the provided name.
    - _select_derivative(self, activation): Selects the derivative of the activation function based on the provided name.
    - _forward(self, X): Performs the forward pass of the neural network.
    - _backward(self, X, y, y_hat, lr): Performs the backward pass of the neural network.
    - train(self, X_train, y_train, X_test, y_test, epochs=25, lr=0.0001, log=True): Trains the neural network model using the provided training data.
    - predict(self, X): Predicts the output for the given input data.
    - parameters(self): Returns the parameters of the neural network.
    - plot(self, history): Plots the training and validation loss over epochs, as well as the accuracy values.
    """
    
    def __init__(self, no_of_features: int, hidden_layer_activation: str, output_layer_activation: str, neurons: int = 32):
        self.neurons = neurons
        self.features = no_of_features
    
        self.hidden_layer_activation = hidden_layer_activation
        self.output_layer_activation = output_layer_activation
        
        self.w1 = np.random.randn(no_of_features, neurons) * 0.01
        self.w2 = np.random.randn(neurons) * 0.01
        self.b1 = np.zeros(neurons)
        self.b2 = 0.0
        
    def __repr__(self):
        return f"Hidden_Layer_NN(no_of_features={self.features}, neurons={self.neurons})"
    
    def summary(self):
        print(f"input shape: {self.features}")
        print(f"hidden layer shape: {self.neurons}")
        print(f"output shape: 1")
    
    def _rmse(self, y, y_hat):
        """
        Calculates the root mean squared error (RMSE) between the predicted values (y_hat) and the actual values (y).

        Parameters:
        - y (numpy.ndarray): The actual values.
        - y_hat (numpy.ndarray): The predicted values.

        Returns:
        - float: The RMSE value.
        """
        error = np.sqrt(np.mean((y - y_hat) ** 2))
        return error
    
    def _accuracy(self, y, y_hat):
        """
        Calculates the accuracy of the predicted values.

        Parameters:
        - y (numpy.ndarray): The true labels.
        - y_hat (numpy.ndarray): The predicted labels.

        Returns:
        - float: The accuracy of the predicted values.
        """
        
        y_pred = np.where(y_hat >= 0.5, 1, 0)
        accuracy  = np.mean(y == y_pred)
        return accuracy
    
    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    def _dsigmoid(self, X):
        return self._sigmoid(X) * (1 - self._sigmoid(X))
    
    def _relu(self, X):
        return np.maximum(X, 0)
    
    def _drelu(self, X):
        return np.where(X > 0, 1, 0)
    
    def _select_activation(self, activation: str):
        """
        Selects the activation function based on the provided name.
        
        Parameters:
        - activation (str): The name of the activation function.
        
        Returns:
        - function: The activation function.
        """
        Activators = {
            "sigmoid": self._sigmoid,
            "relu": self._relu
        }
        return Activators[activation]
    
    def _select_derivative(self, activation: str):
        """
        Selects the derivative of the activation function based on the provided name.
        
        Parameters:
        - activation (str): The name of the activation function.
        
        Returns:
        - function: The derivative of the activation function.
        """
        Derivatives = {
            "sigmoid": self._dsigmoid,
            "relu": self._drelu
        }
        return Derivatives[activation]
    
    def _forward(self, X):
        """
        Performs the forward pass of the neural network.

        Args:
        - X (numpy.ndarray): Input data of shape (n_samples, n_features).

        Returns:
        - numpy.ndarray: Predicted output of shape (n_samples, 1).
        """
        
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self._select_activation(self.hidden_layer_activation)(self.z1)
        self.z2 = np.dot(self.a1, self.w2)
        y_hat = self._select_activation(self.output_layer_activation)(self.z2) + self.b2
        
        return y_hat
    
    def _backward(self, X, y, y_hat, lr):
        """
        Performs the backward pass of the neural network.

        Args:
        - X (numpy.ndarray): Input data.
        - y (numpy.ndarray): Target labels.
        - y_hat (numpy.ndarray): Predicted labels.
        - lr (float): Learning rate.

        Returns:
        - None
        """

        loss = y_hat - y 
        w2_grad = np.dot(self.a1.T, loss * self._select_derivative(self.output_layer_activation)(self.z2))
        a1_grad = np.dot(loss.reshape(-1, 1) * self._select_derivative(self.output_layer_activation)(self.z2.reshape(-1, 1)), self.w2.reshape(-1, 1).T)
        w1_grad = np.dot(X.T, a1_grad * self._select_derivative(self.hidden_layer_activation)(self.z1))
        
        b1_grad = np.mean(a1_grad * self._select_derivative(self.hidden_layer_activation)(self.z1), axis=0)
        b2_grad = np.mean(loss * self._select_derivative(self.output_layer_activation)(self.z2))
        
        self.w1 -= lr * w1_grad
        self.w2 -= lr * w2_grad
        self.b1 -= lr * b1_grad
        self.b2 -= lr * b2_grad
        
    def train(self, X_train, y_train, X_test, y_test, epochs=25, lr=0.0001, log=True):
        """
        Trains the neural network model using the provided training data.

        Parameters:
        - X_train (numpy.ndarray): The input training data.
        - y_train (numpy.ndarray): The target training data.
        - X_test (numpy.ndarray): The input test data.
        - y_test (numpy.ndarray): The target test data.
        - epochs (int): The number of training epochs (default: 25).
        - lr (float): The learning rate for gradient descent (default: 0.0001).
        - log (bool): Whether to log the training progress (default: True).

        Returns:
        - history (dict): A dictionary containing the training history, including the loss and validation loss for each epoch.
        """
        
        print(f"Total Parameters: {self.w1.size + self.w2.size + self.b1.size + 1}")
        print(f"Size of model: {(sys.getsizeof(self.w1) + sys.getsizeof(self.w2) + sys.getsizeof(self.b1) + sys.getsizeof(self.b2))/(1024*1024): .3f} MB")
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

        for epoch in range(1, epochs+1):
            y_hat = self._forward(X_train)
            self._backward(X_train, y_train, y_hat, lr)
            train_pred = self._forward(X_train)
            val_pred = self._forward(X_test)

            loss = self._rmse(y_train, train_pred)
            val_loss = self._rmse(y_test, val_pred)
            accuracy = self._accuracy(y_train, train_pred)
            val_accuracy = self._accuracy(y_test, val_pred)

            history['loss'].append(loss)
            history['val_loss'].append(val_loss)
            history['accuracy'].append(accuracy)
            history['val_accuracy'].append(val_accuracy)

            if log:
                print(f"Epoch {epoch}: \t train_loss: {loss: .3f} \t val_loss: {val_loss: .3f} \t train_accuracy: {accuracy: .3f} \t val_accuracy: {val_accuracy: .3f}")

        return history
    
    def predict(self, X):
        """
        Predicts the output for the given input data.

        Parameters:
        - X (numpy.ndarray): The input data to be predicted.

        Returns:
        - numpy.ndarray: The predicted output for the input data.
        """
        try:
            return self._forward(X)
        except Exception as e:
            print(f"An error occurred: {e}")
            
    def parameters(self):
        """
        Returns the parameters of the neural network.

        Returns:
            dict: A dictionary containing the weights and bias of the neural network.
                - "weights1" (ndarray): The weights of the first layer.
                - "weights2" (ndarray): The weights of the second layer.
                - "bias" (ndarray): The bias of the neural network.
        """
        
        return {"weights": [self.w1, self.w2], "bias": [self.b1, self.b2]}
        
    def save_model(self, path):
        """
        Saves the model to a file.

        Parameters:
        - path (str): The path to save the model.

        Returns:
        - None
        """
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    def load_model(self, path):
        """
        Loads the model from a file.

        Parameters:
        - path (str): The path to load the model from.

        Returns:
        - None
        """
        
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        return model
            
    def save_weights(self, path):
        """
        Saves the weights and bias of the neural network to a file.

        Parameters:
        - path (str): The path to save the weights and bias.

        Returns:
        - None
        """
        
        np.savez(path, w1=self.w1, w2=self.w2, b1=self.b1, b2=self.b2)
        
    def load_weights(self, path):
        """
        Loads the weights and bias of the neural network from a file.

        Parameters:
        - path (str): The path to load the weights and bias.

        Returns:
        - None
        """
        
        data = np.load(path)
        self.w1 = data['w1']
        self.w2 = data['w2']
        self.b1 = data['b1']
        self.b2 = data['b2']
            
    def plot(self, history):
        """
        Plots the training and validation loss over epochs.

        Parameters:
        - history (dict): A dictionary containing the training and validation loss values, as well as the accuracy values.

        Returns:
        - None
        """

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(history['loss'], label='train_loss', color='blue', lw=0.8)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(history['val_loss'], label='val_loss', color='red', lw=0.8)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(history['accuracy'], label='train_accuracy', color='blue', lw=0.8) 
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(history['val_accuracy'], label='val_accuracy', color='red', lw=0.8)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Validation')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=5000, n_features=250, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    model = Single_Layer_NN(no_of_features=X.shape[1], hidden_layer_activation='relu', output_layer_activation='sigmoid', neurons=128)
    history = model.train(X_train, y_train, X_test, y_test, epochs=50, lr=0.0001, log=True)
    model.plot(history)
    