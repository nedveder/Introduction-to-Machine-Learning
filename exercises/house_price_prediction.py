import os

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"
ITERATIONS = 10
START_DATA_PERCENTAGE = 10
END_DATA_PERCENTAGE = 100
CURRENT_YEAR = 2023


def remove_negative_prices(X: pd.DataFrame):
    """
        Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    """
    # Filter out non-positive rows in the 'price' column
    return X[X['price'] > 0]


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # Fill all NA values with mean values of each column
    X.fillna(X.mean(numeric_only=True), inplace=True)

    # If house was renovated and the renovation is new in comparison to other renovations
    X['was_renovated_recently'] = np.where(X['yr_renovated'] >= np.percentile(X['yr_renovated'].unique(), 75), 0, 1)
    X['was_renovated_recently'] = X['was_renovated_recently'].astype('uint8')

    # Deal with zipcode, turn to category and separate
    X['zipcode'] = X['zipcode'].astype('uint32')
    X = pd.get_dummies(X, prefix='zipcode_', columns=['zipcode'])

    # Deal with sqft above and basement, turn to category and separate
    X['percentage_above'] = X['sqft_above'] / (X['sqft_above'] + X['sqft_basement'])
    X['percentage_above'] = X['percentage_above'].astype('float32')

    # Deal with year built, turn to category of how many decades ago was the house built, and separate
    X['decades_ago_built'] = ((CURRENT_YEAR - X['yr_built']) // 10).astype('uint16')
    X = pd.get_dummies(X, prefix='decades_ago_built_', columns=['decades_ago_built'])

    # Remove unwanted features
    X = X.drop(['id', 'date', 'yr_renovated', 'lat', 'long', 'sqft_lot15', 'sqft_living15', 'yr_built', 'sqft_above',
                'sqft_basement'], axis=1)

    # Fix other features
    for col in ['bedrooms', 'bathrooms', 'floors']:
        X[col] = np.where(X[col] >= 0, X[col], X[col].median()).astype('uint16')
        X[col] = np.where(X[col] <= 12, X[col], X[col].median()).astype('uint16')

    for col in ['sqft_living', 'sqft_lot']:
        X[col] = np.where(X[col] > 0, X[col], X[col].median()).astype('uint32')

    X['waterfront'] = X['waterfront'].astype('uint8')
    X['view'] = X['view'].astype('uint8')
    X['condition'] = X['condition'].astype('uint8')
    X['grade'] = X['grade'].astype('uint8')

    return (X, y) if y is not None else X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # iterate over each feature and create a scatter plot with the response variable
    for col in X.columns:
        if 'zipcode_' in col or 'decades_ago_built_' in col:
            continue
        # calculate Pearson correlation between feature and response
        pearson_correlation = np.cov(X[col].values, y.values)[0, 1] / (np.std(X[col]) * np.std(y))

        # create scatter plot
        fig = go.Figure(go.Scatter(x=X[col], y=y, mode='markers'),
                        layout=go.Layout(title=f"{col} (pearson_correlation={pearson_correlation:.2f})",
                                         xaxis=dict(title='Feature'),
                                         yaxis=dict(title='Response')))

        # save plot to file
        if not os.path.exists(f"{output_path}"):
            os.makedirs(f"{output_path}")
        fig.write_image(f'{output_path}/{col}.png')


def generate_regression_figure(p_values, loss_values, std_loss_values):
    fig = go.Figure([go.Scatter(x=p_values, y=loss_values,
                                mode='lines+markers',
                                line=dict(color='black', width=2),
                                name='Average Loss'),
                     go.Scatter(x=p_values, y=loss_values - 2 * std_loss_values,
                                fill=None,
                                mode='lines',
                                line=dict(color='lightgrey'),
                                name='Error Ribbon',
                                showlegend=False),
                     go.Scatter(x=p_values, y=loss_values + 2 * std_loss_values,
                                fill='tonexty',
                                mode='lines',
                                line=dict(color='lightgrey'),
                                name='Confidence Interval')],
                    layout=go.Layout(title='Average Loss as Function of Training Size',
                                     xaxis=dict(title='Training Size (%)'),
                                     yaxis=dict(title='Average Loss')))
    fig.show()


def fit_and_predict_model(p_values, loss_values, std_loss_values):
    """

    """
    linear_regression = LinearRegression(False)
    for i, p in enumerate(p_values):
        loss = np.zeros(ITERATIONS)
        for j in range(ITERATIONS):
            # Create samples
            sample_data_frame = processed_train_x.sample(frac=p / 100)
            sample_data = sample_data_frame.to_numpy(dtype='float64')
            sample_label = processed_train_y[sample_data_frame.index].to_numpy(dtype='float64')
            # Fit model
            linear_regression.fit(sample_data, sample_label)
            # Calculate loss for current training iteration
            loss[j] = linear_regression.loss(processed_test_x.to_numpy(dtype='float64'),
                                             processed_test_y.to_numpy(dtype='float64'))
        # Mean loss over percentage and standard error
        loss_values[i] = np.mean(loss)
        std_loss_values[i] = np.std(loss)
    return loss_values, std_loss_values


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    df = remove_negative_prices(df)
    train_x, train_y, test_x, test_y = split_train_test(df.drop('price', axis=1), df['price'])
    print("Data split into train and test")

    # Question 2 - Preprocessing of housing prices dataset
    processed_train_x, processed_train_y = preprocess_data(train_x), train_y
    processed_test_x, processed_test_y = preprocess_data(test_x), test_y
    print("Data Processed")

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(processed_train_x, processed_train_y, "ex2_plots")
    print("Features evaluated")

    # Question 4 - Fit model over increasing percentages of the overall training data
    n_steps = END_DATA_PERCENTAGE - START_DATA_PERCENTAGE
    training_percentage = np.arange(START_DATA_PERCENTAGE, END_DATA_PERCENTAGE)
    loss_avg_per_batch = np.zeros(n_steps)
    std_loss_avg_per_batch = np.zeros(n_steps)

    loss_avg_per_batch, std_loss_avg_per_batch = fit_and_predict_model(training_percentage,
                                                                       loss_avg_per_batch,
                                                                       std_loss_avg_per_batch)

    # Plot the results
    generate_regression_figure(training_percentage, loss_avg_per_batch, std_loss_avg_per_batch)
    print("Finished Training")
