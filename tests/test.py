import unittest
import numpy as np
from nplearn.linear_model import LinearRegression
from nplearn.preprocessing import StandardScaler
from nplearn.metrics import mean_squared_error

class TestLinearModel(unittest.TestCase):
    def test_linear_regression(self):
        X = np.array([[1], [2], [3], [4]])
        y = np.array([3, 5, 7, 9])
        
        model = LinearRegression()
        model.fit(X, y)
        
        self.assertAlmostEqual(model.coef_[0], 2.0, places=5)
        self.assertAlmostEqual(model.intercept_, 1.0, places=5)
        
        y_pred = model.predict(np.array([[5]]))
        self.assertAlmostEqual(y_pred[0], 11.0, places=5)

class TestPreprocessing(unittest.TestCase):
    def test_standard_scaler(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        self.assertAlmostEqual(np.mean(scaled_data[:, 0]), 0.0, places=5)
        
if __name__ == '__main__':
    unittest.main()