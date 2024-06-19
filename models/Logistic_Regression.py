import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

class LogisticRegressor:
    """
    A simple Logistic Regressor class for binary classification tasks. The Logistic Regressor has a single layer with sigmoid activation.
    
    Attributes:
    - features: int, the number of features in the input data
    - w: numpy array, the weights of the Logistic Regressor
    - b: float, the bias of the Logistic Regressor
    
    Methods:
    - __init__ (no_of_features) : Initializes the Logistic Regressor with random weights and bias
    - _forward (X) : Calculates the predicted value for the given input
    - _backward (X, y, y_hat, lr) : Calculates the gradient and updates the weights and bias
    - train (X_train, y_train, X_test, y_test, epochs=25, lr=0.0001) : Trains the Logistic Regressor
    - predict (X) : Predicts the target values for the given input data
    - parameters () : Returns the weights and bias of the Logistic Regressor
    - save_model (path) : Saves the model to a file
    - load_model (path) : Loads the model from a file
    - save_weights (path) : Saves the weights and bias of the model to a file
    - load_weights (path) : Loads the weights and bias of the model from a file
    - plot (history) : Plots the training and validation loss and accuracy history
    """
    
    def __init__(self, no_of_features: int):
        """
        - Initializes the Logistic Regressor with random weights and bias.
        """
        
        self.features = no_of_features
        self.w = np.random.randn(no_of_features) * 0.01
        self.b = 0.0
        
    def __repr__(self):
        return f"LogisticRegressor(no_of_features={self.features})"
    
    def _forward(self, X):
        """
        Calculates the predicted value for the given input.

        Args:
        - X: numpy array, the input data
        Returns:
        - numpy array, the predicted value
        """
        
        z = np.dot(X, self.w) + self.b
        return 1 / (1 + np.exp(-z))
    
    def _backward(self, X, y, y_hat, lr) -> None:
        """
        Calculates the gradient and updates the weights and bias.

        Args:
        - X: numpy array, the input data
        - y: numpy array, the target values
        - y_hat: numpy array, the predicted values
        - lr: float, the learning rate
        
        Returns:
        - None
        """
        
        m = X.shape[0]  

        dw = np.dot(X.T, y_hat - y) / m
        db = np.sum(y_hat - y) / m
        
        self.w -= lr * dw
        self.b -= lr * db
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs=25, lr=0.0001, log=True):
        """
        Trains the Logistic Regressor.
        
        Args:
        - X_train: numpy array, the training input data
        - y_train: numpy array, the training target values
        - X_test: numpy array, the validation input data
        - y_test: numpy array, the validation target values
        - epochs: int, the number of epochs
        - lr: float, the learning rate
        - log: bool, whether to log the training progress
        
        Returns:
        - dict, the training history
        """
        
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        
        for epoch in range(1, epochs+1):
            y_hat = self._forward(X_train)
            self._backward(X_train, y_train, y_hat, lr)
            train_pred = self._forward(X_train)
            val_pred = self._forward(X_test)
            
            loss = -np.mean(y_train * np.log(train_pred) + (1 - y_train) * np.log(1 - train_pred))
            val_loss = -np.mean(y_test * np.log(val_pred) + (1 - y_test) * np.log(1 - val_pred))
            accuracy = np.mean((train_pred >= 0.5) == y_train)
            val_accuracy = np.mean((val_pred >= 0.5) == y_test)
            
            history['loss'].append(loss)
            history['val_loss'].append(val_loss)
            history['accuracy'].append(accuracy)
            history['val_accuracy'].append(val_accuracy)
            
            if log:
                print(f"Epoch {epoch}: \t train_loss: {loss: .4f} \t val_loss: {val_loss: .4f} \t train_accuracy: {accuracy: .4f} \t val_accuracy: {val_accuracy: .4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for the given input data.
        
        Args:
        - X: numpy array, the input data
        
        Returns:
        - numpy array, the predicted values
        """
        return self._forward(X)
    
    def parameters(self) -> dict:
        """
        Returns the weights and bias of the Logistic Regressor.
        
        Returns:
        - dict, the weights and bias
        """
        return {"weights": self.w, "bias": self.b}
    
    def save_model(self, path: str) -> None:    
        """
        Saves the model to a file.
        
        Args:
        - path: str, the path to save the model
        
        Returns:
        - None
        """
        
        with open(path, 'wb') as file:
            pkl.dump(self, file)
            
    def load_model(self, path: str) -> 'LogisticRegressor':
        """
        Loads the model from a file.
        
        Args:
        - path: str, the path to load the model from
        
        Returns:
        - LogisticRegressor, the loaded model
        """
        
        with open(path, 'rb') as file:
            model = pkl.load(file)
            
        return model
    
    def save_weights(self, path: str) -> None:
        """
        Saves the weights and bias of the model to a file.
        
        Args:
        - path: str, the path to save the weights and bias
        
        Returns:
        - None
        """
        with open(path, 'wb') as file:
            pkl.dump(self.parameters(), file)
            
    def load_weights(self, path: str) -> None:
        """
        Loads the weights and bias of the model from a file.
        
        Args:
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
        Plots the training and validation loss and accuracy history.
        
        Args:
        - history: dict, the training history
        
        Returns:
        - None
        """
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(history['loss'], label='train_loss', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(history['val_loss'], label='val_loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Validation')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(history['accuracy'], label='train_accuracy', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(history['val_accuracy'], label='val_accuracy', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Validation')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(n_samples=2000, n_features=100, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = LogisticRegressor(no_of_features=X_train.shape[1])
    history = model.train(X_train, y_train, X_test, y_test, epochs=5000, lr=0.001, log=True)
    model.plot(history)