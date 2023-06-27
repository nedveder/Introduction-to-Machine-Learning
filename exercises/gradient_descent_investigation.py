import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from sklearn.metrics import roc_curve

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def callback(**kwargs):
        values.append(kwargs["val"])
        weights.append(kwargs["weights"])

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    figures = []
    norms = []
    for eta in etas:
        out_option_figures = []
        out_option_norms = []
        for out_type in ["last", "best", "average"]:
            l1, l2 = L1(init), L2(init)
            call_l1, v_l1, w_l1, call_l2, v_l2, w_l2 = *get_gd_state_recorder_callback(), *get_gd_state_recorder_callback()
            optimizer1, optimizer2 = GradientDescent(FixedLR(eta), callback=call_l1, out_type=out_type), \
                GradientDescent(FixedLR(eta), callback=call_l2, out_type=out_type)
            optimizer1.fit(l1, X=np.array([]), y=np.array([]))
            optimizer2.fit(l2, X=np.array([]), y=np.array([]))
            out_option_figures.append((plot_descent_path(L1, np.array(w_l1)),
                                       plot_descent_path(L2, np.array(w_l2))))
            out_option_norms.append((v_l1, v_l2))
        norms.append(out_option_norms[0])
        figures.append(out_option_figures[0])

    fig = make_subplots(rows=2, cols=4, subplot_titles=[f"eta={eta}" for eta in etas])
    fig2 = go.Figure()
    for i, (out_option_figures, out_option_norms) in enumerate(zip(figures, norms), start=1):
        fig.add_trace(out_option_figures[0].data[0], row=1, col=i)
        fig.add_trace(out_option_figures[0].data[1], row=1, col=i)
        fig.add_trace(out_option_figures[1].data[0], row=2, col=i)
        fig.add_trace(out_option_figures[1].data[1], row=2, col=i)

        fig2.add_trace(go.Scatter(x=list(range(len(out_option_norms[0]))), y=out_option_norms[0],
                                  name=f"L1 with eta={etas[i - 1]}"))
        fig2.add_trace(go.Scatter(x=list(range(len(out_option_norms[1]))), y=out_option_norms[1],
                                  name=f"L2 with eta={etas[i - 1]}"))

    fig.update_layout(title="GD Descent Path for different fixed learning rates L1 (top) and L2 (bottom)",
                      showlegend=False, margin=dict(t=100))
    fig2.update_layout(title="Objective value for different fixed learning rates (Convergence rate)",
                       legend_title="Learning Rate",
                       showlegend=True, margin=dict(t=100), xaxis_title="Iteration", yaxis_title="Objective value")
    fig.show()
    fig2.show()
    # Print the final objective value for each model
    print(
        f"Lowest objective value for L1 with fixed learning rates:{ min([val for norm in norms for val in norm[0]])}")
    print(
        f"Lowest objective value for L2 with fixed learning rates:{ min([val for norm in norms for val in norm[1]])}")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    figures = []
    for gamma in gammas:
        l1 = L1(init)
        call_l1, _, w_l1 = get_gd_state_recorder_callback()
        optimizer1 = GradientDescent(ExponentialLR(base_lr=eta, decay_rate=gamma), callback=call_l1)
        optimizer1.fit(l1, X=np.array([]), y=np.array([]))
        figures.append((plot_descent_path(L1, np.array(w_l1), title=f"gamma={gamma}")))

    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"gamma={eta}" for eta in gammas])
    for i, figure in enumerate(figures):
        fig.add_trace(figure.data[0], row=i // 2 + 1, col=i % 2 + 1)
        fig.add_trace(figure.data[1], row=i // 2 + 1, col=i % 2 + 1)
    fig.update_layout(title="GD Descent Path for different for different exponential decay rates",
                      showlegend=False, margin=dict(t=100))
    fig.show()

    # Plot descent path for gamma=0.95
    fig = figures[1]
    fig.update_layout(title="GD Descent Path for gamma=0.95", showlegend=False, margin=dict(t=100))
    fig.show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train.values, y_train.values)
    # Plot ROC curve using predict_proba method
    probas = logistic_regression.predict_proba(X_test.values)
    fpr, tpr, thresholds = roc_curve(y_test.values, probas)
    fig = go.Figure([go.Scatter(x=fpr, y=tpr, mode="lines")],
                    layout=go.Layout(xaxis=dict(title="False Positive Rate"),
                                     yaxis=dict(title="True Positive Rate"),
                                     title=f"ROC curve For alpha = {thresholds[np.argmax(tpr - fpr)]}"
                                           f" the TPR-FPR difference is maximized"))
    fig.show()

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    from IMLearn.model_selection import cross_validate
    from IMLearn.metrics import misclassification_error

    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    train_scores_l1, validation_scores_l1, train_scores_l2, validation_scores_l2 = [], [], [], []
    for penalty in [("l1", train_scores_l1, validation_scores_l1), ("l2", train_scores_l2, validation_scores_l2)]:
        for lam in lambdas:
            model = LogisticRegression(penalty=penalty[0], alpha=.5, lam=lam)
            model.solver_.max_iter_ = 20000
            model.solver_.learning_rate_ = FixedLR(1e-4)
            train_score, validation_score = cross_validate(model, X_train.values, y_train.values,
                                                           misclassification_error)
            penalty[1].append(train_score)
            penalty[2].append(validation_score)

    best_l1_lambda = lambdas[np.argmin(validation_scores_l1)]
    best_l2_lambda = lambdas[np.argmin(validation_scores_l2)]
    fig = go.Figure([go.Scatter(x=lambdas, y=train_scores_l1, mode="lines", name="Train L1"),
                     go.Scatter(x=lambdas, y=validation_scores_l1, mode="lines", name="Validation L1"),
                     go.Scatter(x=lambdas, y=train_scores_l2, mode="lines", name="Train L2"),
                     go.Scatter(x=lambdas, y=validation_scores_l2, mode="lines", name="Validation L2"),
                     go.Scatter(x=[best_l1_lambda], y=[min(validation_scores_l1)], mode="markers",
                                name=f"Best L1 lambda: {best_l1_lambda}"),
                     go.Scatter(x=[best_l2_lambda], y=[min(validation_scores_l2)], mode="markers",
                                name=f"Best L2 lambda: {best_l2_lambda}")],
                    layout=go.Layout(xaxis=dict(title="lambda"),
                                     yaxis=dict(title="Misclassification error"),
                                     title="Misclassification error for different values of lambda"))
    fig.show()
    l1_logistic_regression = LogisticRegression(penalty="l1", alpha=.5, lam=best_l1_lambda)
    l1_logistic_regression.fit(X_train.values, y_train.values)
    print(f"Best L1 lambda: {best_l1_lambda},"
          f" Model's test error: {l1_logistic_regression.loss(X_test.values, y_test.values)}")
    l2_logistic_regression = LogisticRegression(penalty="l2", alpha=.5, lam=best_l2_lambda)
    l2_logistic_regression.fit(X_train.values, y_train.values)
    print(f"Best L2 lambda: {best_l2_lambda},"
          f" Model's test error: {l2_logistic_regression.loss(X_test.values, y_test.values)}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
