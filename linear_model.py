import numpy as np

class LinearRegression():
    def __init__(self):
        self.theta = None
        self.intercept_ = None
        self.coef_ = None
    
    def fit(self, X : np.ndarray, y  : np.ndarray) -> None:
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))
        XTX = X_b.T @ X_b
        XTy = X_b.T @ y
        self.theta = np.linalg.solve(XTX, XTy)
        self.intercept_ = self.theta[0, 0]
        self.coef_ = self.theta[1:, 0]
        return self

    def predict(self, X : np.ndarray) -> np.array:
        if self.theta is None:
            raise ValueError("LinearRegression needs data to fit. Please use **.fit(self, X : np.array, y : np.array) to let the regressor to calculate parameters first.")
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))
        y_pred = X_b @ self.theta
        return y_pred.flatten()
    
    def score(self, X : np.ndarray, y : np.ndarray) -> float:
        if self.theta is None:
            raise ValueError("LinearRegression needs data to fit. Please use **.fit(self, X : np.array, y : np.array) to let the regressor to calculate parameters first.")
        if y.ndim == 1:
            y_true = y.reshape(-1, 1)
        else:
            y_true = y
        y_pred = self.predict(X).reshape(-1, 1)
        y_mean = np.mean(y_true)
        SS_tot = np.sum((y_true - y_mean) ** 2)
        SS_res = np.sum((y_true - y_pred) ** 2)
        R2 = 1 - (SS_res / SS_tot)
        return R2

class RidgeRegression(LinearRegression):
    def __init__(self, alpha : float = 1.0):
        super().__init__()
        self.alpha = alpha

    def fit(self, X : np.ndarray = None, y : np.ndarray = None) -> None:
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))
        I_prime = np.eye(X_b.shape[1])
        I_prime[0, 0] = 0
        XTX = X_b.T @ X_b
        XTy = X_b.T @ y
        A = XTX + self.alpha * I_prime
        self.theta = np.linalg.solve(A, XTy)
        self.intercept_ = self.theta[0, 0]
        self.coef_ = self.theta[1:, 0]
        return self
    
    def predict(self, X : np.ndarray) -> np.ndarray:
        if self.theta is None:
            raise ValueError("RidgeRegression needs data to fit. Please use **.fit(self, X : np.array, y : np.array) to let the regressor to calculate parameters first.")
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))

        y_pred = X_b @ self.theta

        return y_pred
    
    def score(self, X, y):
        return super().score(X, y)
    

class LassoRegression(LinearRegression):
    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-4):
        super().__init__()
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X : np.ndarray = None, y : np.ndarray = None) -> None:
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_samples, n_features = X.shape
        self.coef_ = np.zeros((n_features, 1))
        self.intercept_ = np.mean(y)
        for iteration in range(self.max_iter):
            old_coef = self.coef_.copy()
            for j in  range(n_features):
                y_pred = self.intercept_ + X @ self.coef_
                residual = y - y_pred + X[:, j].reshape(-1, 1) * self.coef_[j]
                z = np.sum(X[:, j].reshape(-1, 1) * residual)
                denominator = np.sum(X[:, j] ** 2)
                if denominator == 0:
                    new_coef_j = 0.0
                else:
                    gamma = self.alpha
                    
                    if z > gamma:
                        new_coef_j = (z - gamma) / denominator
                    elif z < -gamma:
                        new_coef_j = (z + gamma) / denominator
                    else:
                        new_coef_j = 0.0
                self.coef_[j] = new_coef_j
            self.intercept_ = np.mean(y - X @ self.coef_)
            coef_change = np.linalg.norm(self.coef_ - old_coef)
            if coef_change < self.tol:
                break
        return self
    
    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("LassoRegression needs data to fit. Please use **.fit(self, X : np.array, y : np.array) to let the regressor to calculate parameters first.")
        y_pred = self.intercept_ + X @ self.coef_
        return y_pred.flatten()

    def score(self, X, y):
        return super().score(X, y)
