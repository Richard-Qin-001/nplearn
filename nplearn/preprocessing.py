import numpy as np
from typing import Union, Optional, Tuple

class StandardScaler():
    def __init__(self, *args, **kwargs):
        self.mean_ : Optional[np.ndarray] = None
        self.scale_ : Optional[np.ndarray] = None
        self.epsilon = 1e-8

    def fit(self, X : np.ndarray) -> 'StandardScaler':
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        return self
    
    def transform(self, X : np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before calling transform.")
        zero_scale_mask = (self.scale_ < self.epsilon)
        divisor = np.where(zero_scale_mask, 1.0, self.scale_)
        X_scaled = (X - self.mean_) / divisor
        X_scaled[:, zero_scale_mask] = 0.0
        return X_scaled
    
    def fit_transform(self, X : np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_scaled : np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before calling inverse_transform.")
        X_original = X_scaled * self.scale_ + self.mean_
        return X_original
    
class MinMaxScaler():
    """
    Scale features to a specified range (default is [0, 1]).
    """
    def __init__(self, feature_range : Tuple[float, float] = (0, 1), *args, **kwargs):
        if not isinstance(feature_range, tuple) or len(feature_range) != 2:
            raise TypeError("feature_range must be a tuple of (min, max).")
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired range must be less than the maximum.")
        self.feature_range = feature_range
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
    
    def fit(self, X : np.ndarray) -> "MinMaxScaler":
        """
        Calculate the minimum and maximum value of each feature in the training set X.

        Parameters:
        X (np.ndarray): The data to be fitted.

        Returns:
        MinMaxScaler: The object itself.
        """
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)

        return self
    
    def transform(self, X : np.ndarray) -> np.ndarray:
        """
        Scale the data using the previously computed min_ and max_.

        Parameters:
        X (np.ndarray): The data to be transformed.

        Returns:
        np.ndarray: The scaled data.
        """
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before calling transform")
        
        data_range = self.max_ - self.min_
        target_min, target_max = self.feature_range
        target_range = target_max - target_min
        data_range = np.where(data_range < 1e-8, 1.0, data_range)
        X_norm = (X - self.min_) / data_range
        X_scaled = X_norm * target_range + target_min
        constant_features = (self.max_ - self.min_) < 1e-8
        X_scaled[:, constant_features] = target_min
        return X_scaled
            
    def fit_transform(self, X : np.ndarray) -> np.ndarray:
        """
        First fit on X, then transform X.

        
        Parameters:
        X (np.ndarray): The data to be transformed.

        Returns:
        np.ndarray: The scaled data.
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_scaled : np.ndarray) -> np.ndarray:
        """
        Restore the scaled data back to its original scale.

        Parameters:
        X_scaled (np.ndarray): The scaled data to be inverse transformed.

        Returns:
        np.ndarray: The original data
        """
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before calling inverse_transform")
        
        data_range = self.max_ - self.min_

        target_min, target_max = self.feature_range
        target_range = target_max - target_min

        target_range_inv = np.where(target_range < 1e-8, 1.0, target_range)
        X_norm = (X_scaled - target_min) / target_range_inv
        X_original = X_norm * data_range + self.min_

        return X_original
