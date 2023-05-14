from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    fig = make_subplots(rows=1, cols=2)
    for i, (n, f) in enumerate(
            [("Linearly Separable", "linearly_separable.npy"),
             ("Linearly Inseparable", "linearly_inseparable.npy")]):
        # Load dataset
        X, true_y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        Perceptron(callback=lambda perceptron, _, __: losses.append(
            perceptron.loss(X, true_y))).fit(X, true_y)

        # Plot figure of loss as function of fitting iteration
        fig.add_trace(go.Scatter(y=losses, mode='lines', name=n), row=1,
                      col=i + 1)

    fig.update_layout(title='Perceptron Algorithm Training Loss',
                      xaxis_title='Training Iterations',
                      yaxis_title='Loss')
    fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1", "gaussian2"]:
        # Load dataset
        X, true_y = load_dataset(f"../datasets/{f}.npy")

        # Fit models and predict over training set
        lda = LDA()
        gnb = GaussianNaiveBayes()
        lda.fit(X, true_y)
        gnb.fit(X, true_y)
        lda_pred, gnb_pred = lda.predict(X), gnb.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes
        # predictions on the left and LDA predictions on the right.
        # Plot title should specify dataset used and subplot titles should
        # specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(f'Gaussian Naive Bayes Accuracy: {100 * accuracy(gnb_pred, true_y):.2f}%',
                                            f'LDA Accuracy: {100 * accuracy(lda_pred, true_y):.2f}%'))

        # Add traces for data-points. Color represents predicted class, marker symbol represents actual class
        color_palette = ["cornflowerblue", "indianred", "burlywood"]  # color palette
        marker_symbols = ['circle', 'triangle-up', 'cross']  # marker symbols

        for i, pred in enumerate([gnb_pred, lda_pred], start=1):
            marker_dict = dict(color=[color_palette[int(p)] for p in pred],
                               symbol=[marker_symbols[int(ty)] for ty in true_y])
            fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=marker_dict), row=1, col=i)

        # Add `X` dots specifying fitted Gaussians' means
        for i, model in enumerate([gnb, lda], start=1):
            fig.add_trace(go.Scatter(x=model.mu_[:, 0], y=model.mu_[:, 1], mode="markers",
                                     marker=dict(symbol="x", color="black")), row=1, col=i)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(np.unique(true_y))):
            fig.add_trace(get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i])), row=1, col=1)
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1, col=2)

        # Update layout and show figure
        fig.update_layout(title_text=f'Comparing Classifiers on {f} Dataset', showlegend=False)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
