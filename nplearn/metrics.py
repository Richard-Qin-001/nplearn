import numpy as np
from typing import Optional

def mean_squared_error(
        y_true : np.ndarray, y_pred : np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        multioutput: str = 'uniform_average') -> float:
    """
    Calculate Mean Squared Error (MSE), supporting sample weights and multi-output averaging.
    """
    if not y_true.shape == y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    squared_error = (y_true - y_pred) ** 2
    if sample_weight is not None:
        if sample_weight.shape != y_true.shape[0]:
            raise ValueError("sample_weight must be of shape (n_samples,).")
        if y_true.ndim > 1:
            sample_weight = sample_weight.reshape(-1, 1)
        sum_output = np.sum(sample_weight * squared_error, axis=0)
        n_samples = np.sum(sample_weight)
    else:
        sum_output = np.sum(squared_error, axis=0)
        n_samples = y_true.shape[0]
    
    if y_true.ndim > 1:
        mse_per_output = sum_output / n_samples
        return np.mean(mse_per_output)
    else:
        return sum_output / n_samples

def root_mean_squared_error(
        y_true : np.ndarray, y_pred : np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        multioutput: str = 'uniform_average') -> float:
    mse = mean_squared_error(y_true, y_pred, sample_weight, multioutput)
    rmse = np.sqrt(mse)
    return rmse

def mean_squared_log_error(
        y_true : np.ndarray, y_pred : np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        multioutput: str = 'uniform_average') -> float:
    if not y_true.shape == y_pred.shape:
        raise TypeError("y_true and y_pred must have the same shape")
    if y_true.ndim == 1:
        y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred.reshape(-1, 1)
    msle = np.mean((np.log(y_true + 1) - np.log(y_pred + 1)) ** 2)
    return msle
    
def root_mean_squared_log_error(
        y_true : np.ndarray, y_pred : np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        multioutput: str = 'uniform_average') -> float:
    msle = mean_squared_log_error(y_true, y_pred)
    rmsle = np.sqrt(msle)
    return rmsle

def mean_absolute_error(
        y_true : np.ndarray, y_pred : np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        multioutput: str = 'uniform_average') -> float:
    if not y_true.shape == y_pred.shape:
        raise TypeError("y_true and y_pred must have the same shape")
    if y_true.ndim == 1:
        y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred.reshape(-1, 1)
    mae = np.mean(np.abs(y_true - y_pred))

    return mae

def r2_score(
        y_true : np.ndarray, y_pred : np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        multioutput: str = 'uniform_average') -> float:
    """
    Calculate the R^2 score, supporting sample weights and multi-output averaging.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if sample_weight is not None:
        if sample_weight.shape != y_true.shape[0]:
            raise ValueError("sample_weight must be of shape (n_samples,).")
        if y_true.ndim > 1:
            sample_weight = sample_weight.reshape(-1, 1)
        
        y_mean = np.average(y_true, axis=0, weights=sample_weight)
        SS_tot_per_output = np.sum(sample_weight * (y_true - y_mean) ** 2, axis=0)
        
        SS_res_per_output = np.sum(sample_weight * (y_true - y_pred) ** 2, axis=0)
    else:
        y_mean = np.mean(y_true, axis=0)
        SS_tot_per_output = np.sum((y_true - y_mean) ** 2, axis=0)
        SS_res_per_output = np.sum((y_true - y_pred) ** 2, axis=0)
    
    non_zero_ss_tot = SS_tot_per_output != 0
    if np.any(non_zero_ss_tot):
        R2_per_output = np.ones_like(SS_tot_per_output)
        R2_per_output[non_zero_ss_tot] = 1 - (SS_res_per_output[non_zero_ss_tot] / SS_tot_per_output[non_zero_ss_tot])
    else:
        R2_per_output = np.where(SS_res_per_output == 0, 1.0, 0.0)

    if y_true.ndim > 1:
        return np.mean(R2_per_output)
    else:
        return R2_per_output.item()

def median_absolute_error(
        y_true : np.ndarray, y_pred : np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        multioutput: str = 'uniform_average') -> float:
    if not y_true.shape == y_pred.shape:
        raise TypeError("y_true and y_pred must have the same shape")
    if y_true.ndim == 1:
        y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred.reshape(-1, 1)
    medae = np.median(np.abs(y_true - y_pred))
    return medae

def mean_absolute_percentage_error(
        y_true : np.ndarray, y_pred : np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        multioutput: str = 'uniform_average') -> float:
    if not y_true.shape == y_pred.shape:
        raise TypeError("y_true and y_pred must have the same shape")
    if y_true.ndim == 1:
        y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred.reshape(-1, 1)
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    return mape

def max_error(
        y_true : np.ndarray, y_pred : np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        multioutput: str = 'uniform_average') -> float:
    if not y_true.shape == y_pred.shape:
        raise TypeError("y_true and y_pred must have the same shape")
    if y_true.ndim == 1:
        y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred.reshape(-1, 1)
    max_err = np.max(np.abs(y_true - y_pred))
    return max_err

def mean_gamma_deviance(
        y_true : np.ndarray, y_pred : np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        multioutput: str = 'uniform_average') -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if np.any(y_true <= 0) or np.any(y_pred <= 0):
        raise ValueError("y_true and y_pred must be strictly positive for Gamma deviance.")
    
    deviance = 2 * (np.log(y_pred / y_true) + (y_true - y_pred) / y_pred)
    if sample_weight is not None:
        if sample_weight.shape != y_true.shape[0]:
            raise ValueError("sample_weight must be of shape (n_samples,).")
        if y_true.ndim > 1:
            sample_weight = sample_weight.reshape(-1, 1)
        return np.sum(sample_weight * deviance) / np.sum(sample_weight)
    else:
        return np.mean(deviance)

def mean_poisson_deviance(
        y_true : np.ndarray, y_pred : np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        multioutput: str = 'uniform_average') -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    y_pred[y_pred <= 0] = np.finfo(y_pred.dtype).eps
    deviance = 2 * (y_true * np.log(y_true / y_pred) - (y_true - y_pred))
    deviance[y_true == 0] = 2 * y_pred[y_true == 0]
    if sample_weight is not None:
        if sample_weight.shape != y_true.shape[0]:
            raise ValueError("sample_weight must be of shape (n_samples,).")
        if y_true.ndim > 1:
            sample_weight = sample_weight.reshape(-1, 1)

        return np.sum(sample_weight * deviance) / np.sum(sample_weight)
    else:
        return np.mean(deviance)

def explained_variance_score(
        y_true : np.ndarray, y_pred : np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        multioutput: str = 'uniform_average') -> float:
    if not y_true.shape == y_pred.shape:
        raise TypeError("y_true and y_pred must have the same shape")
    if y_true.ndim == 1:
        y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred.reshape(-1, 1)
    evs = 1 - np.var(y_true - y_pred) / np.var(y_true)
    return evs
