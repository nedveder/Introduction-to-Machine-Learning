import os

import tqdm
from matplotlib import pyplot as plt

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

pio.templates.default = "simple_white"
CURRENT_YEAR = 2023


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    # Remove Nan values and duplicate rows
    X.dropna(inplace=True)
    X.drop_duplicates(inplace=True)
    # Remove proper from y as well
    if y is not None:
        y = y[X.index]

    # Filter out negative rows in the proper columns
    excluded_columns = ['lat', 'long']
    for col in X.columns.difference(excluded_columns):
        if col == 'date':
            negative_indices = X['date'] != 0
        elif col == 'bedrooms':
            negative_indices = (X[col] > 0) & (X[col] < 20)
        elif col == 'bathrooms':
            negative_indices = (X[col] > 0)
        else:
            negative_indices = X[col] >= 0

        X = X.loc[negative_indices]
        if y is not None:
            y = y.loc[negative_indices]

    # Assign proper types for each column
    dtypes_dict = {'bedrooms': np.dtype('uint8'),
                   'bathrooms': np.dtype('float16'),
                   'sqft_living': np.dtype('uint16'),
                   'sqft_lot': np.dtype('uint32'),
                   'floors': np.dtype('float16'),
                   'waterfront': np.dtype('bool'),
                   'view': np.dtype('uint8'),
                   'condition': np.dtype('uint8'),
                   'grade': np.dtype('uint8'),
                   'sqft_above': np.dtype('uint16'),
                   'sqft_basement': np.dtype('uint16'),
                   'yr_built': np.dtype('uint16'),
                   'yr_renovated': np.dtype('uint16'),
                   'lat': np.dtype('float32'),
                   'long': np.dtype('float32'),
                   'sqft_living15': np.dtype('uint16'),
                   'sqft_lot15': np.dtype('uint16')}
    X = X.astype(dtypes_dict)

    # Add and remove features
    X['is_renovated'] = X['yr_renovated'] != 0

    # Remove unused features
    X = X.drop(['id', 'date', 'yr_built', 'yr_renovated', 'condition', 'long'], axis=1)

    X['zipcode'] = X['zipcode'].astype('category')
    if y is None:
        return X

    # Filter out non-positive rows in the 'price' column
    negative_price_indices = y > 0
    X = X[negative_price_indices]
    y = y[negative_price_indices]

    return X, y


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
    for col in X.columns.difference(['zipcode']):
        # calculate Pearson correlation between feature and response
        pearson_correlation = np.cov(X[col].values, y.values)[0, 1] / (np.std(X[col]) * np.std(y))

        # create scatter plot
        ax = sns.scatterplot(x=X[col], y=y)
        ax.set_title(f"{col} (pearson_correlation={pearson_correlation:.2f})")

        # save plot to file
        if not os.path.exists(f"{output_path}"):
            os.makedirs(f"{output_path}")
        plt.savefig(f'{output_path}/{col}.png')
        plt.close()


def visualize_data(X):
    fig, axes = plt.subplots(6, 3, figsize=(12, 10))
    ax = axes.flatten()

    for i, col in enumerate(X.columns.difference(['zipcode', 'long', 'lat'])):
        sns.histplot(X[col], ax=ax[i], color='lightblue')  # histogram call
        ax[i].set_title(col)  # add title
        ax[i].text(0.03, 0.03, f"Max: {X[col].max()}\nMin: {X[col].min()}", weight='bold',
                   transform=ax[i].transAxes)  # add max and min values to corner of plot

    sns.jointplot(x=X['long'], y=X['lat'], kind='hist')

    fig.tight_layout(w_pad=2, h_pad=1)  # change padding
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")
    # Question 1 - split data into train and test sets
    train_x, train_y, test_x, test_y = split_train_test(df.drop('price', axis=1), df['price'])

    # Question 2 - Preprocessing of housing prices dataset
    processed_train_x, processed_train_y = preprocess_data(train_x, train_y)
    processed_test_x, processed_test_y = preprocess_data(test_x, test_y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(processed_train_x, processed_train_y, "ex2_plots")

    # Question 4 - Fit model over increasing percentages of the overall training data
    p_values = np.array(range(10, 101, 1))
    loss_values = np.zeros(len(p_values))
    std_loss_values = np.zeros(len(p_values))

    for i, p in tqdm.tqdm(enumerate(p_values)):
        n_samples = int(len(processed_train_x) * (p / 100))
        loss = np.zeros(10)
        for iteration in range(10):
            sample_data_frame = processed_train_x.sample(n=n_samples)
            sample_data = sample_data_frame.to_numpy(dtype='float64')
            sample_label = processed_train_y[sample_data_frame.index].to_numpy(dtype='float64')

            linear_regression = LinearRegression()
            linear_regression.fit(sample_data, sample_label)

            loss[iteration] = linear_regression.loss(processed_test_x.to_numpy(dtype='float64'),
                                                     processed_test_y.to_numpy(dtype='float64'))
        loss_values[i] = np.mean(loss)
        std_loss_values[i] = np.std(loss)

    # Plot the results
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
                                     xaxis_title='Training Size (%)',
                                     yaxis_title='Average Loss'))
    fig.show()
