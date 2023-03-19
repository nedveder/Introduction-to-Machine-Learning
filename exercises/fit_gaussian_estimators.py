import plotly

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(10, 1, 1000)
    univariate_gaussian = UnivariateGaussian()
    univariate_gaussian.fit(X)
    print(univariate_gaussian.mu_, univariate_gaussian.var_)

    # Question 2 - Empirically showing sample mean is consistent
    # TODO LEFT HERE
    estimate_mean = [(univariate_gaussian.fit(X[:current_sample_index]).mu_, current_sample_index)
                     for current_sample_index in range(10, 1000, 10)]
    fig = px.line(estimate_mean, x="Sample Size", y="| Estimated Mean - True Mean |",
                  title="Difference between Estimated mean and True mean per Sample size")
    fig.show()
    # Question 3 - Plotting Empirical PDF of fitted model
    raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
