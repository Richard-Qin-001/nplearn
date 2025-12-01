import numpy as np
from typing import Optional, List
from .base import BaseEstimator, RegressorMimin

class LinearRegression(BaseEstimator, RegressorMimin):
    def __init__(self):
        self.theta = None
        self.intercept_ = None
        self.coef_ = None
    
    def fit(self, X : np.ndarray, y  : np.ndarray) -> 'LinearRegression':
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

    def predict(self, X : np.ndarray) -> np.ndarray:
        if self.theta is None:
            raise ValueError("LinearRegression needs data to fit. Please use **.fit(self, X : np.array, y : np.array) to let the regressor to calculate parameters first.")
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))
        y_pred = X_b @ self.theta
        return y_pred.flatten()
    
    # def score(self, X : np.ndarray, y : np.ndarray) -> float:
    #     if self.theta is None:
    #         raise ValueError("LinearRegression needs data to fit. Please use .fit(X : np.array, y : np.array) to let the regressor to calculate parameters first.")
    #     if y.ndim == 1:
    #         y_true = y.reshape(-1, 1)
    #     else:
    #         y_true = y
    #     y_pred = self.predict(X).reshape(-1, 1)
    #     y_mean = np.mean(y_true)
    #     SS_tot = np.sum((y_true - y_mean) ** 2)
    #     SS_res = np.sum((y_true - y_pred) ** 2)
    #     R2 = 1 - (SS_res / SS_tot)
    #     return R2
    def score(self, X : np.ndarray, y : np.ndarray) -> float:
        return super().score(X, y)

class RidgeRegression(LinearRegression):
    def __init__(self, alpha : float = 1.0):
        super().__init__()
        self.alpha = alpha

    def fit(self, X : np.ndarray = None, y : np.ndarray = None) -> 'RidgeRegression':
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
            raise ValueError("RidgeRegression needs data to fit. Please use .fit(X : np.array, y : np.array) to let the regressor to calculate parameters first.")
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))

        y_pred = X_b @ self.theta

        return y_pred.flatten()
    
    def score(self, X : np.ndarray, y : np.ndarray) -> float:
        return super().score(X, y)
    

class LassoRegression(LinearRegression):
    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-4):
        super().__init__()
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X : np.ndarray = None, y : np.ndarray = None) -> 'LassoRegression':
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
            self.intercept_ = np.mean(y - X @ self.coef_).item()
            coef_change = np.linalg.norm(self.coef_ - old_coef)
            if coef_change < self.tol:
                break
        return self
    
    def predict(self, X : np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("LassoRegression needs data to fit. Please use .fit(X : np.array, y : np.array) to let the regressor to calculate parameters first.")
        y_pred = self.intercept_ + X @ self.coef_
        return y_pred.flatten()

    def score(self, X : np.ndarray, y : np.ndarray) -> float:
        return super().score(X, y)

class ElasticNet(LassoRegression):
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 1000, tol: float = 1e-4, *args, **kwargs):
        super().__init__(alpha=alpha, max_iter=max_iter, tol=tol)
        self.l1_ratio = l1_ratio
        if not (0.0 <= l1_ratio <= 1.0):
             raise ValueError("l1_ratio must be between 0.0 and 1.0.")
    def fit(self, X : np.ndarray, y : np.ndarray) -> "ElasticNet":
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_samples, n_features = X.shape
        self.coef_ = np.zeros((n_features, 1))
        self.intercept_ = np.mean(y).item()
        lambda_l1 = self.alpha * self.l1_ratio
        lambda_l2 = self.alpha * (1.0 - self.l1_ratio)
        for iteration in range(self.max_iter):
            old_coef = self.coef_.copy()
            self.intercept_ = np.mean(y - X @ self.coef_).item()

            for j in range(n_features):
                y_pred = self.intercept_ + X @ self.coef_
                residual = y - y_pred + X[:, j].reshape(-1, 1) * self.coef_[j]
                z = np.sum(X[:, j].reshape(-1, 1) * residual)
                denominator = np.sum(X[:, j] ** 2) + lambda_l2
                
                if denominator == 0:
                    new_coef_j = 0.0
                else:
                    z_prime = z
                    if z_prime > lambda_l1:
                        new_coef_j = (z_prime - lambda_l1) / denominator
                    elif z_prime < -lambda_l1:
                        new_coef_j = (z_prime + lambda_l1) / denominator
                    else:
                        new_coef_j = 0.0
                
                self.coef_[j] = new_coef_j
            coef_change = np.linalg.norm(self.coef_ - old_coef)
            if coef_change < self.tol:
                break
        return self
    
    def predict(self, X : np.ndarray) -> np.ndarray:
        return super().predict(X)
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return super().score(X, y)

class HuberRegression(LinearRegression):
    def __init__(self, epsilon: float = 1.35, max_iter: int = 500, tol: float = 1e-4, learning_rate: float = 0.01, *args, **kwargs):
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be strictly positive.")
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
    
    def _huber_derivative(self, residuals : np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the Huber loss function with respect to the residuals.
        """
        inner_mask = np.abs(residuals) <= self.epsilon
        outer_mask = np.abs(residuals) > self.epsilon

        derivative = np.zeros_like(residuals, dtype=float)

        derivative[inner_mask] = residuals[inner_mask]

        derivative[outer_mask] = self.epsilon * np.sign(residuals[outer_mask])

        return derivative
    
    def fit(self, X : np.ndarray, y : np.ndarray) -> "HuberRegression":
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_samples, n_features = X.shape

        self.theta = np.zeros((n_features + 1, 1))
        ones = np.ones((n_samples, 1))
        X_b = np.hstack((ones, X))

        for iteration in range(self.max_iter):
            old_theta = self.theta.copy()

            y_pred = X_b @ self.theta
            residuals = y - y_pred
            dL_dres = self._huber_derivative(residuals)

            gradient = - (X_b.T @ dL_dres)

            self.theta = self.theta - self.learning_rate * gradient

            theta_change = np.linalg.norm(self.theta - old_theta)
            if theta_change < self.tol:
                print(f"HuberRegression converged after {iteration+1} iterations.")
                break
        self.intercept_ = self.theta[0, 0]
        self.coef_ = self.theta[1:, 0]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return super().predict(X)
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return super().score(X, y)

class Lars(LinearRegression):
    def __init__(self, max_features: Optional[int] = None, eps: float = 1e-10):
        super().__init__()
        self.max_feaatures = max_features
        self.eps = eps
        self.coef_path_ = None

    def fit(self, X : np.ndarray, y : np.ndarray) -> "Lars":
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_samples, n_features = X.shape

        self.intercept_ = np.mean(y).item()
        y_centered = y - self.intercept_
        self.coef_ = np.zeros((n_features, 1))
        self.coef_path_ = [self.coef_.copy().flatten()]
        active_set = set()
        residuals = y_centered

        k = 0
        while len(active_set) < n_features and (self.max_feaatures is None or k < self.max_feaatures):
            k += 1
            correlations = X.T @ residuals
            C_max = np.max(np.abs(correlations))
            A_candidates = np.where(np.abs(correlations) >= C_max - self.eps)[0]
            new_active_features = [i for i in A_candidates if i not in active_set]

            if new_active_features:
                j_new = new_active_features[0]
                active_set.add(j_new)
            
            A_indices = sorted(list(active_set))
            X_A = X[:, A_indices]

            s_A = np.sign(correlations[A_indices])

            G_A = X_A.T @ X_A

            try:
                G_inv = np.linalg.inv(G_A)
            except np.linalg.LinAlgError:
                return self
            
            A_g = G_inv @ s_A

            A_u = X_A @ A_g

            gamma_star = C_max / np.sum(A_g @ s_A)
            gamma_hat = np.inf

            inactive_indices = [i for i in range(n_features) if i not in active_set]

            for j in inactive_indices:
                x_j = X[:, j].reshape(-1, 1)
                a_j = x_j @ A_u
                C_j = correlations[j].item()
                a_j_val = a_j.item()

                if np.abs(a_j_val - C_max) > self.eps:
                    gamma1 = (C_max - C_j) / (C_max - a_j_val)
                    if gamma1 > self.eps:
                        gamma_hat = min(gamma_hat, gamma1)
                if np.abs(a_j_val + C_max) > self.eps:
                    gamma2 = (C_max + C_j) / (C_max + a_j_val)
                    if gamma2 > self.eps:
                        gamma_hat = min(gamma_hat, gamma2)
            final_gamma = min(gamma_hat, gamma_star)
            coef_update = final_gamma * A_g.flatten()
            for idx, i in enumerate(A_indices):
                self.coef_[i] += coef_update[idx]
            
            residuals = residuals - final_gamma * A_u

            self.coef_path_.append(self.coef_.copy().flatten())

            if np.abs(final_gamma - gamma_hat) < self.eps:
                pass
            if np.abs(final_gamma - gamma_star) < self.eps:
                pass

        self.coef_ = self.coef_.reshape(-1, 1)

        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("LeastAngleRegression needs data to fit.")
        y_pred = X @ self.coef_ + self.intercept_

        return y_pred.flatten()
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return super().score(X, y)

class LassoLars(Lars):
    def fit(self, X : np.ndarray, y : np.ndarray) -> "LassoLars":
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_samples, n_features = X.shape

        self.intercept_ = np.mean(y).item()
        y_centered = y - self.intercept_
        self.coef_ = np.zeros((n_features, 1))
        self.coef_path_ = [self.coef_.copy().flatten()]
        active_set :List[int] = []

        residuals = y_centered
        k = 0

        while len(active_set) < n_features and (self.max_feaatures is None or k < self.max_feaatures):
            k += 1

            correlations = X.T @ residuals
            C_max = np.max(np.abs(correlations))
            
            if not active_set:
                j_new = np.argmax(np.abs(correlations))
                active_set.append(j_new)
            else:
                A_candidates = np.where(np.abs(correlations) >= C_max - self.eps)[0]
                new_features = [i for i in A_candidates if i not in active_set]
                if new_features:
                    active_set.append(new_features[0])
            
            X_A = X[:, active_set]
            s_A = np.sign(correlations[active_set])
            G_A = X_A.T @ X_A
            
            try:
                G_inv = np.linalg.inv(G_A)
            except np.linalg.LinAlgError:
                return self
            
            A_g = G_inv @ s_A
            A_u = X_A @ A_g

            gamma_star = C_max / np.sum(A_g @ s_A)
            gamma_hat = np.inf

            inactive_indices = [i for i in range(n_features) if i not in active_set]

            for j in inactive_indices:
                x_j = X[:, j].reshape(-1, 1)
                a_j = x_j @ A_u
                C_j = correlations[j].item()
                a_j_val = a_j.item()

                if np.abs(a_j_val - C_max) > self.eps:
                    gamma1 = (C_max - C_j) / (C_max - a_j_val)
                    if gamma1 > self.eps:
                        gamma_hat = min(gamma_hat, gamma1)
                if np.abs(a_j_val + C_max) > self.eps:
                    gamma2 = (C_max + C_j) / (C_max + a_j_val)
                    if gamma2 > self.eps:
                        gamma_hat = min(gamma_hat, gamma2)
            
            gamma_zero = np.inf
            coef_A = self.coef_[active_set].flatten()
            for idx, i in enumerate(active_set):
                if np.abs[A_g[i]] > self.eps:
                    gamma_step = -coef_A[idx] / A_g[idx].flatten()
                    if gamma_step > self.eps:
                        gamma_zero = min(gamma_zero, gamma_step)
            final_gamma = min(gamma_star, gamma_hat, gamma_zero)
            coef_update = final_gamma * A_g.flatten()
            for idx, i in enumerate(active_set):
                self.coef_[i] += coef_update[idx]
            
            residuals = residuals - final_gamma * A_u

            self.coef_path_.append(self.coef_.copy().flatten())

            if np.abs(final_gamma - gamma_hat) < self.eps:
                zero_indices = [i for i in active_set if np.abs(self.coef_[i]) < self.eps]
                for i in zero_indices:
                    active_set.remove(i)
                continue
            if np.abs(final_gamma - gamma_star) < self.eps:
                break

        self.coef_ = self.coef_.reshape(-1, 1)

        return self

class OrthogonalMatchingPursuit(LinearRegression):
    """
    Orthogonal Matching Pursuit (OMP) Algorithm
    """
    def __init__(self, n_nonzero_coefs: Optional[int] = None, tol: Optional[float] = None, eps : float = 1e-10):
        super().__init__()
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.eps = eps
    def fit(self, X : np.ndarray, y : np.ndarray) -> "OrthogonalMatchingPursuit":
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_samples, n_features = X.shape

        self.intercept_ = np.mean(y).item()
        y_centered = y - self.intercept_

        active_set : List[int] = []
        residuals = y_centered.copy()
        K = self.n_nonzero_coefs
        if K is None:
            K = n_features
        elif K > n_features:
            raise ValueError("K must be less than the number of features.")
        self.coef_ = np.zeros((n_features, 1))
        for k in range(K):
            correlations = X.T @ residuals
            abs_correlations = np.abs(correlations)

            for i in active_set:
                abs_correlations[i] = 0.0

            j_new = np.argmax(abs_correlations)
            if abs_correlations[j_new] < self.eps:
                break
            active_set.append(j_new)
            X_A = X[:, active_set]
            try:
                w_A = np.linalg.solve(X_A.T @ X_A, X_A.T @ y_centered)
            except np.linalg.LinAlgError:
                raise ValueError("Matrix is singular.")
                break
            y_pred_A = X_A @ w_A
            residuals = y_centered - y_pred_A

        for idx, i in enumerate(active_set):
            self.coef_[i] = w_A[idx]

        return self
    def predict(self, X : np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("OrthogonalMatchingPursuit needs data to fit.")
        y_pred = X @ self.coef_ + self.intercept_        
        return y_pred.flatten()

class BayesianRidge(LinearRegression):
    def __init__(self):
        super().__init__()