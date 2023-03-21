import pandas as pd
import plotly

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import tqdm

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    TRUE_MEAN = 10
    TRUE_VAR = 1
    NUMBER_OF_SAMPLES = 1000

    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(TRUE_MEAN, TRUE_VAR, size=NUMBER_OF_SAMPLES)
    univariate_gaussian = UnivariateGaussian()
    univariate_gaussian.fit(X)
    print(univariate_gaussian.mu_, univariate_gaussian.var_)

    # Question 2 - Empirically showing sample mean is consistent
    samples_mean = np.array([
        (abs(TRUE_MEAN - univariate_gaussian.fit(X[:current_sample_index]).mu_), current_sample_index)
        for current_sample_index in tqdm.trange(10, NUMBER_OF_SAMPLES, 10)])

    df1 = pd.DataFrame(samples_mean, columns=["Mean difference", "Sample size"])

    fig_q2 = px.line(df1, range_x=(0, NUMBER_OF_SAMPLES), x="Sample size", range_y=(0, 0.8), y="Mean difference"
                     , markers=True, title="Difference between Estimated mean and True mean over sample size")
    fig_q2.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    df2 = pd.DataFrame(np.array([X, univariate_gaussian.pdf(X)]).T, columns=["Value", "PDF value"])
    fig_q3 = px.scatter(df2, x="Value", range_y=(0, 0.5), y="PDF value", title="PDF Value over sample values")
    fig_q3.show()


def test_multivariate_gaussian():
    TRUE_MEAN = np.array([0, 0, 4, 0])
    TRUE_COV_MAT = np.array([[1, 0.2, 0, 0.5],
                             [0.2, 2, 0, 0],
                             [0, 0, 1, 0],
                             [0.5, 0, 0, 1]])
    NUMBER_OF_SAMPLES = 1000
    # Question 4 - Draw samples and print fitted model
    X = np.random.multivariate_normal(TRUE_MEAN, TRUE_COV_MAT, size=NUMBER_OF_SAMPLES)
    multivariate_gaussian = MultivariateGaussian()
    multivariate_gaussian.fit(X)
    print(multivariate_gaussian.mu_)
    print(multivariate_gaussian.cov_)
    # Question 5 - Likelihood evaluation
    NUMBER_OF_TRIALS = 200

    f1, f3 = np.linspace(-10, 10, NUMBER_OF_TRIALS), np.linspace(-10, 10, NUMBER_OF_TRIALS)

    log_likelihood = np.array(
        [[multivariate_gaussian.log_likelihood(np.array([t1, 0, t2, 0]).T, TRUE_COV_MAT, X) for t1 in f1]
         for t2 in tqdm.tqdm(f3)])

    fig = px.imshow(log_likelihood, title="Log Likelihood over Features 1 and 3",
                    labels=dict(x="Feature 3", y="Feature 1", color="Log likelihood"), x=f3, y=f1
                    , color_continuous_scale=px.colors.sequential.Inferno)
    fig.update_xaxes(side="top")
    fig.show()
    # Question 6 - Maximum likelihood
    f1_index, f3_index = np.unravel_index(log_likelihood.argmax(), log_likelihood.shape)
    print(round(f1[f1_index], 4), round(f3[f3_index], 4))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
