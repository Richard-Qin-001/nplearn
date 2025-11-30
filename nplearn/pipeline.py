import numpy as np
from typing import List, Tuple, Any

class Pipeline():
    def __init__(self, steps : List[Tuple[str, Any]],*args, **kwargs):
        if not isinstance(steps, list) or not all(isinstance(s, tuple) and len(s) == 2 for s in steps):
            raise ValueError("Steps must be a list of (name, estimator) tuples.")
        self.steps = steps
        if not self.steps:
            raise ValueError("Pipeline must contain at least one step.")
        self.estimator = self.steps[-1][1]
    
    def fit(self, X : np.ndarray, y : np.ndarray) -> 'Pipeline':
        current_X = X

        for name, transformer in  self.steps[:-1]:
            if hasattr(transformer, 'fit_transform'):
                current_X = transformer.fit_transform(current_X)
            elif hasattr(transformer, 'fit') and hasattr(transformer, 'transform'):
                transformer.fit(current_X)
                current_X = transformer.transform(current_X)
            else:
                 raise TypeError(f"Transformer '{name}' does not implement fit_transform or fit/transform.")
        
        self.estimator.fit(current_X, y)
        return self
    
    def predict(self, X : np.ndarray) -> np.ndarray:
        current_X = X
        for name, step in self.steps:
            if step is self.estimator:
                if hasattr(step, 'predict'):
                    return step.predict(current_X)
                else:
                    raise TypeError(f"Estimator '{name}' does not implement predict.")
            else:
                if hasattr(step, 'transform'):
                    current_X = step.transform(current_X)
                else:
                    raise TypeError(f"Transformer '{name}' does not implement transform.")
        return current_X
    
    def score(self, X : np.ndarray, y : np.ndarray) -> float:
        y_pred = self.predict(X)
        if hasattr(self.estimator, 'score'):
            if y.ndim == 1:
                y_true = y.reshape(-1, 1)
            else:
                y_true = y
            y_pred = y_pred.reshape(-1, 1)
            y_mean = np.mean(y_true)
            SS_tot = np.sum((y_true - y_mean) ** 2)
            SS_res = np.sum((y_true - y_pred) ** 2)
            if SS_tot == 0:
                return 1.0
            R2 = 1 - (SS_res / SS_tot)
            return R2
        else:
            raise TypeError("Estimator does not implement score method")