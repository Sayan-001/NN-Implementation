import numpy as np
import unittest

from Linear_Regressor import LinearRegressor

class TestLinearRegressor(unittest.TestCase):
    def setUp(self):
        self.X_train = np.array([[1.0, 2, 3], [4, 5.0, 6], [7, 8, 9.0], [10.0, 11.0, 12], [13, 14.0, 15.0]])
        self.y_train = np.array([10, 20.5, 30, 40.8, 50])
        self.X_test = np.array([[2.0, 3.0, 4], [5.0, 6, 7.0], [8.0, 9.0, 10]])
        self.y_test = np.array([15, 25.3, 35.8])
        self.model = LinearRegressor(self.X_train.shape[1])

    def test_forward(self):
        y_pred = self.model.forward(self.X_train)
        self.assertEqual(y_pred.shape, (self.X_train.shape[0],), msg="The shape of the predicted values is incorrect")

    def test_backward(self):
        y_pred = self.model.forward(self.X_train)
        self.model.backward(self.X_train, self.y_train, y_pred, lr=0.001)
        self.assertEqual(self.model.w.shape, (self.X_train.shape[1],))
        self.assertIsInstance(self.model.b, float, msg="The bias should be a float")

    def test_train(self):
        history = self.model.train(self.X_train, self.y_train, self.X_test, self.y_test, epochs=5, lr=0.001, log=False)
        self.assertEqual(len(history['loss']), 5, msg="The length of the training loss history is incorrect")
        self.assertEqual(len(history['val_loss']), 5, msg="The length of the validation loss history is incorrect")

    def test_predict(self):
        y_pred = self.model.predict(self.X_test)
        self.assertEqual(y_pred.shape, (self.X_test.shape[0],), msg="The shape of the predicted values is incorrect")

if __name__ == '__main__':
    unittest.main()