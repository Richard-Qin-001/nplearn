import numpy as np
from .base import BaseEstimator, RegressorMixin
from typing import Optional

def xlogy(x: np.ndarray, y: np.ndarray, out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_y = np.log(y)
        result = np.where(x == 0, 0.0, x * log_y)

    if out is not None:
        np.copyto(out, result)
        return None
    else:
        return result


def inplace_identity(X : np.ndarray):
    """Simply leave the input array unchanged.

    Parameters
    ----------
    X : np.ndarray
    """
    # Nothing to do

def inplace_exp(X : np.ndarray):

    np.exp(X, out=X)

def inplace_logistic(X : np.ndarray):

    X = 1 / (1 + np.exp(-X))

def inplace_tanh(X : np.ndarray):

    np.tanh(X, out=X)

def inplace_relu(X : np.ndarray):

    np.maximum(X, 0, out=X)

def inplace_softmax(X : np.ndarray):
    
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]

ACTIVATIONS = {
    "identity": inplace_identity,
    "exp": inplace_exp,
    "tanh": inplace_tanh,
    "logistic": inplace_logistic,
    "relu": inplace_relu,
    "softmax": inplace_softmax,
}

def inplace_identity_derivative(Z : np.ndarray, delta : np.ndarray):
    """Apply the derivative of the identity function: do nothing.

    Parameters
    ----------
    Z : {np.ndarray}, shape (n_samples, n_features)
        The data which was output from the identity activation function during
        the forward pass.

    delta : {np.ndarray}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    # Nothing to do

def inplace_exp_derivative(Z : np.ndarray, delta : np.ndarray):
    """Apply the derivative of the exponential function: do nothing.

    Parameters
    ----------
    Z : {np.ndarray}, shape (n_samples, n_features)
        The data which was output from the identity activation function during
        the forward pass.

    delta : {np.ndarray}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    # Nothing to do

def inplace_logistic_derivative(Z : np.ndarray, delta : np.ndarray):
    """Apply the derivative of the logistic sigmoid function.

    It exploits the fact that the derivative is a simple function of the output
    value from logistic function.

    Parameters
    ----------
    Z : {np.ndarray}, shape (n_samples, n_features)
        The data which was output from the identity activation function during
        the forward pass.

    delta : {np.ndarray}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta *= Z
    delta *= 1 - Z

def inplace_tanh_derivative(Z : np.ndarray, delta : np.ndarray):
    """Apply the derivative of the hyperbolic tanh function.

    It exploits the fact that the derivative is a simple function of the output
    value from hyperbolic tangent.

    Parameters
    ----------
    Z : {np.ndarray}, shape (n_samples, n_features)
        The data which was output from the identity activation function during
        the forward pass.

    delta : {np.ndarray}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta *= 1 - Z**2

def inplace_relu_derivative(Z : np.ndarray, delta : np.ndarray):
    """Apply the derivative of the relu function.

    It exploits the fact that the derivative is a simple function of the output
    value from rectified linear units activation function.

    Parameters
    ----------
    Z : {np.ndarray}, shape (n_samples, n_features)
        The data which was output from the identity activation function during
        the forward pass.

    delta : {np.ndarray}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta[Z==0] = 0

DERIVATIVES = {
    "identity": inplace_identity_derivative,
    "tanh": inplace_tanh_derivative,
    "logistic": inplace_logistic_derivative,
    "relu": inplace_relu_derivative,
}

def squared_loss(y_true : np.ndarray, y_pred : np.ndarray, sample_weight=None):
    """Compute the squared loss for regression.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) values.

    y_pred : array-like or label indicator matrix
        Predicted values, as returned by a regression estimator.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    return (
        0.5 * np.average((y_true - y_pred) ** 2, weights=sample_weight, axis=0).mean()
    )


def poisson_loss(y_true : np.ndarray, y_pred : np.ndarray, sample_weight=None):
    """Compute (half of the) Poisson deviance loss for regression.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.

    y_pred : array-like or label indicator matrix
        Predicted values, as returned by a regression estimator.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    # TODO: Decide what to do with the term `xlogy(y_true, y_true) - y_true`. For now,
    # it is included. But the _loss module doesn't use it (for performance reasons) and
    # only adds it as return of constant_to_optimal_zero (mainly for testing).
    return np.average(
        xlogy(y_true, y_true / y_pred) - y_true + y_pred, weights=sample_weight, axis=0
    ).sum()


def log_loss(y_true, y_prob, sample_weight=None):
    """Compute Logistic loss for classification.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.

    y_prob : array-like of float, shape = (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    eps = np.finfo(y_prob.dtype).eps
    y_prob = np.clip(y_prob, eps, 1 - eps)
    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    return -np.average(xlogy(y_true, y_prob), weights=sample_weight, axis=0).sum()


def binary_log_loss(y_true, y_prob, sample_weight=None):
    """Compute binary logistic loss for classification.

    This is identical to log_loss in binary classification case,
    but is kept for its use in multilabel case.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.

    y_prob : array-like of float, shape = (n_samples, 1)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    eps = np.finfo(y_prob.dtype).eps
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.average(
        xlogy(y_true, y_prob) + xlogy(1 - y_true, 1 - y_prob),
        weights=sample_weight,
        axis=0,
    ).sum()


LOSS_FUNCTIONS = {
    "squared_error": squared_loss,
    "poisson": poisson_loss,
    "log_loss": log_loss,
    "binary_log_loss": binary_log_loss,
}


class MLPRegressor():
    pass

class MLPClassifier():
    pass

class BernoulliRBM():
    pass