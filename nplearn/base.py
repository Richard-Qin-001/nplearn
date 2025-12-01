import numpy as np
from typing import Dict, Any, Optional
import inspect

class BaseEstimator():
    def get_params(self, deep : bool = True) -> Dict[str, Any]:
        """
        Get the hyperparameters of the estimator.

        Parameters:
        deep (bool): If True, will recursively get the parameters of sub-estimators contained within the estimator.

        Returns:
        Dict[str, Any]: A dictionary of the estimator's parameter names and their values.
        """
        init_signature = inspect.signature(self.__init__)
        parameters = [p for p in init_signature.parameters.values() if p.name != "self" and p.kind != p.VAR_KEYWORD]
        params = {}
        for p in parameters:
            value = getattr(self, p.name, None)
            params[p.name] = value
        if deep:
            pass
        return params
    
    def set_params(self, **params : Any) -> "BaseEstimator":
        """
        Set the hyperparameters of the estimator.

        Parameters:
        **params: The names and values of the parameters to set.

        Returns:
        BaseEstimator: Returns self.
        """
        if not params:
            return self
        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}. ")
            setattr(self, key, value)
        return self
class TransformerMimin():
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params: Any) -> np.ndarray:
        """
        First fit on X, then transform X.

        
        Parameters:
        X (np.ndarray): The data to be transformed.

        Returns:
        np.ndarray: The scaled data.
        """
        if y is not None:
            return self.fit(X, **fit_params).transform(X, **fit_params)
        else:
            return self.fit(X, y, **fit_params).transform(X, **fit_params)

class RegressorMimin():
    def score(self, X : np.ndarray, y: np.ndarray) -> float:
        from .metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)