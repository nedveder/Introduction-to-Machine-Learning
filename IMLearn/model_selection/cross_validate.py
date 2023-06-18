from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # Split data into folds
    X_folds, y_folds = np.array_split(X, cv), np.array_split(y, cv)
    # Initialize scores
    train_scores = 0
    validation_scores = 0
    # Loop over folds
    for i in range(cv):
        # Create train and validation sets
        train_X, train_y = deepcopy(X_folds), deepcopy(y_folds)
        validation_X, validation_y = train_X.pop(i), train_y.pop(i)
        train_X, train_y = np.concatenate(train_X), np.concatenate(train_y)
        # Fit estimator
        estimator.fit(train_X, train_y)
        # Compute scores
        train_scores += scoring(train_y, estimator.predict(train_X))
        validation_scores += scoring(validation_y, estimator.predict(validation_X))
    return train_scores / cv, validation_scores / cv
