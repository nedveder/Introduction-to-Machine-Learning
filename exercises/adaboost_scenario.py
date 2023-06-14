import numpy as np
from typing import Tuple
from IMLearn.metalearners import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    ada = AdaBoost(DecisionStump, n_learners)
    ada.fit(train_X, train_y)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    train_loss = [ada.partial_loss(train_X, train_y, t) for t in range(1, n_learners + 1)]
    test_loss = [ada.partial_loss(test_X, test_y, t) for t in range(1, n_learners + 1)]
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=np.arange(1, n_learners + 1), y=train_loss, mode='lines', name='Train loss'), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=np.arange(1, n_learners + 1), y=test_loss, mode='lines', name='Test loss'), row=1,
                  col=1)
    fig.update_layout(title=f'Noise ratio: {noise}', xaxis_title='Number of learners', yaxis_title='Loss')
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    symbols = np.where(test_y == 1, "circle", "x")
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\text{{{t} Classifiers in ensemble}}$" for t in T])
    for i, t in enumerate(T):
        fig.add_traces(
            [decision_surface(predict_t(ada, t), lims[0], lims[1], density=60, showscale=False),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                        marker=dict(color=test_y, symbol=symbols))],
            rows=i // 2 + 1, cols=i % 2 + 1)
    fig.update_layout(title="Decision surfaces of AdaBoost ensembles", height=800, width=800)
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_t = np.argmin(test_loss) + 1
    fig = make_subplots(rows=1, cols=1)
    fig.add_traces(
        [decision_surface(predict_t(ada, best_t), lims[0], lims[1], density=60, showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                    marker=dict(color=test_y, symbol=symbols))], rows=1, cols=1)
    fig.update_layout(title=f'Best ensemble size: {best_t}, Accuracy: {1 - test_loss[best_t - 1]:.2f}', width=800,
                      height=800)
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 4: Decision surface with weighted samples
    D = ada.D_ / ada.D_.max() * 5
    fig = go.Figure([decision_surface(ada.predict, lims[0], lims[1], density=60, showscale=False),
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                marker=dict(size=D, color=train_y, symbol=symbols, line=dict(width=0.5), opacity=0.5))])
    fig.update_layout(width=800, height=800, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      title="AdaBoost Decision Surface with weighted samples")
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()


def predict_t(ada, t):
    """
    Returns a function that predicts the label of a sample using the first t classifiers in the ensemble
    """
    return lambda X: ada.partial_predict(X, t)


if __name__ == '__main__':
    np.random.seed(0)
    for noise in [0, 0.4]:
        fit_and_evaluate_adaboost(noise)
