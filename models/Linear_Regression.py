import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

class LinearRegressor:
    """
    A simple Linear Regressor class for regression tasks. The Linear Regressor has a single layer with linear activation.

    Attributes:
    - w: numpy array, the weights of the Linear Regressor
    - b: float, the bias of the Linear Regressor

    Methods:
    - __init__ (input_dim) : Initializes the Linear Regressor with random weights and bias
    - forward (X) : Calculates the predicted value for the given input
    - backward (X, y, y_hat, lr) : Calculates the gradient and updates the weights and bias
    - train (X_train, y_train, X_test, y_test, epochs=25, lr=0.0001) : Trains the Linear Regressor
    - predict (X) : Predicts the target values for the given input data
    - save_model (path) : Saves the model to a file
    - load_model (path) : Loads the model from a file
    - save_weights (path) : Saves the weights and bias of the model to a file
    - load_weights (path) : Loads the weights and bias of the model from a file
    - plot_loss (history) : Plots the training and validation loss history
    """

    def __init__(self, no_of_features: int):
        """
        Initializes the Linear Regressor with random weights and bias.

        Parameters:
        - input_dim: int, the dimension of the input data
        """
        
        self.w = np.random.randn((no_of_features)) * 0.01
        self.b = 0.0
        
    def __repr__(self):
        return f"NN(input_dim={self.w.shape[0]})"
    
    def parameters(self) -> dict:
        """
        Returns the weights and bias of the Linear Regressor.

        Returns:
        - dictionary, the weights and bias
        """
        
        return {"weights": self.w, "bias": self.b}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the predicted value for the given input.

        Parameters:
        - X: numpy array, the input data

        Returns:
        - numpy array, the predicted value
        """
        
        return np.dot(X, self.w) + self.b

    def backward(self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray, lr: float) -> None:
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

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int = 25, lr: float = 0.0001, log: bool = True) -> dict:
        """
        Trains the Regressor.

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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
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
            
    def save_model(self, path: str) -> None:
        """
        Saves the model to a file.

        Parameters:
        - path: str, the path to save the model
        
        Returns:
        - None
        """
        
        with open(path, 'wb') as file:
            pkl.dump(self, file)
            
    def load_model(self, path: str):
        """
        Loads the model from a file.

        Parameters:
        - path: str, the path to load the model from
        """
        
        with open(path, 'rb') as file:
            model = pkl.load(file)

        return model
    
    def save_weights(self, path: str) -> None:
        """
        Saves the weights and bias of the model to a file.

        Parameters:
        - path: str, the path to save the weights and bias
        
        Returns:
        - None
        """
        
        with open(path, 'wb') as file:
            pkl.dump(self.parameters(), file)
            
    def load_weights(self, path: str) -> None:
        """
        Loads the weights and bias of the model from a file.

        Parameters:
        - path: str, the path to load the weights and bias from
        
        Returns:
        - None
        """
        
        with open(path, 'rb') as file:
            weights = pkl.load(file)
            
        self.w = weights['weights']
        self.b = weights['bias']
    
    def plot(self, history: dict) -> None:
        """
        Plots the training and validation loss history.

        Parameters:
        - history: dict, a dictionary containing the training and validation loss history

        Returns:
        - None
        """
        
        plt.figure(figsize=(10, 4))

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
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    X, y = make_regression(n_samples=5000, n_features=500, n_informative=100, noise=0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    model = LinearRegressor(X.shape[1])
    history = model.train(X_train, y_train, X_test, y_test, epochs=50, lr=0.0001, log=True)
    model.plot(history)