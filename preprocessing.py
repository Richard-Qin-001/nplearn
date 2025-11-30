import numpy as np
from typing import Union, Optional

class StandardScaler():
    def __init__(self, *args, **kwargs):
        self.mean_ : Optional[np.ndarray] = None
        self.scale_ : Optional[np.ndarray] = None

    def fit(self, X : np.ndarray) -> 'StandardScaler':
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0

        return self
    
    def transform(self, X : np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before calling transform.")
        X_scaled = (X - self.mean_) / self.scale_
        return X_scaled
    
    def fit_transform(self, X : np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_scaled : np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before calling inverse_transform.")
        X_original = X_scaled * self.scale_ + self.mean_
        return X_original