from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_diabetes

from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    load_diabetes()
    x, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y, test_X, test_y = x[:n_samples], y[:n_samples], x[n_samples:], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_labmdas, lasso_labmdas = np.linspace(0, 0.5, n_evaluations), np.linspace(0, 2.5, n_evaluations)
    ridge_score, lasso_score = np.zeros((n_evaluations, 2)), np.zeros((n_evaluations, 2))
    for i, (ridge_lam, lasso_lam) in enumerate(zip(ridge_labmdas, lasso_labmdas)):
        ridge_score[i] = cross_validate(RidgeRegression(ridge_lam), train_X, train_y, mean_square_error)
        lasso_score[i] = cross_validate(Lasso(lasso_lam), train_X, train_y, mean_square_error)

    # Plot results for Ridge and Lasso and compare train- and validation errors
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Ridge", "Lasso"))
    fig.add_traces([
        go.Scatter(x=ridge_labmdas, y=ridge_score[:, 0], name="Train"),
        go.Scatter(x=ridge_labmdas, y=ridge_score[:, 1], name="Validation")
    ], rows=1, cols=1)
    fig.add_traces([
        go.Scatter(x=lasso_labmdas, y=lasso_score[:, 0], name="Train"),
        go.Scatter(x=lasso_labmdas, y=lasso_score[:, 1], name="Validation")
    ], rows=1, cols=2)
    fig.update_layout(title="Ridge and Lasso Cross-Validation", xaxis_title="Regularization Parameter",
                      yaxis_title="Mean Square Error")
    fig.show()
    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    optimal_ridge, optimal_lasso = ridge_labmdas[np.argmin(ridge_score[:, 1])], lasso_labmdas[np.argmin(lasso_score[:, 1])]
    ridge_model, lasso_model = RidgeRegression(optimal_ridge), Lasso(optimal_lasso, max_iter=5000)
    ridge_model.fit(train_X, train_y)
    lasso_model.fit(train_X, train_y)
    least_squares_model = LinearRegression()
    least_squares_model.fit(train_X, train_y)
    print(f"Best Ridge Model: {optimal_ridge}, Best Lasso Model: {optimal_lasso}")
    print(f"Ridge Train Error: {mean_square_error(ridge_model.predict(train_X), train_y)}")
    print(f"Ridge Test Error: {mean_square_error(ridge_model.predict(test_X), test_y)}")
    print(f"Lasso Train Error: {mean_square_error(lasso_model.predict(train_X), train_y)}")
    print(f"Lasso Test Error: {mean_square_error(lasso_model.predict(test_X), test_y)}")
    print(f"Least Squares Train Error: {mean_square_error(least_squares_model.predict(train_X), train_y)}")
    print(f"Least Squares Test Error: {mean_square_error(least_squares_model.predict(test_X), test_y)}")


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
