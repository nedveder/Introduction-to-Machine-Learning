import os

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

CATEGORICAL_COLUMNS = ['zipcode', 'decades_ago_built', 'grade', 'view', 'condition', 'bedrooms', 'floors']

pio.templates.default = "simple_white"
ITERATIONS = 10
START_DATA_PERCENTAGE = 10
END_DATA_PERCENTAGE = 100
CURRENT_YEAR = 2023
train_columns = []
training_data_column_median = []


def remove_na_prices(X: pd.DataFrame):
    """
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    """
    # Filter out non-positive rows in the 'price' column
    return X[X['price'].notna()]


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
    global training_data_column_median
    is_test = y is None

    if not is_test:
        training_data_column_median = X.median(numeric_only=True)

    # Fill all NA values with median values of each column
    X.fillna(value=training_data_column_median, inplace=True)

    # If house was renovated recently
    X['was_renovated_recently'] = (X['yr_renovated'] > training_data_column_median.T['yr_renovated'])
    # Deal with zipcode
    X['zipcode'] = X['zipcode'].astype('uint32')
    # Deal with year built, turn to category of how many decades ago was the house built
    X['decades_ago_built'] = ((CURRENT_YEAR - X['yr_built'].astype('uint16')) // 10).astype('uint16')
    X['waterfront'] = X['waterfront'].astype('uint8')

    X = preprocess_categorical(X, is_test)

    # Remove unwanted features
    X = X.drop(['id', 'date', 'yr_renovated', 'lat', 'long', 'sqft_lot15', 'sqft_living15', 'yr_built'], axis=1)

    # Fix other features
    X['bathrooms'] = np.where(X['bathrooms'] >= 0, X['bathrooms'], training_data_column_median.T['bathrooms'])
    X['bathrooms'] = np.where(X['bathrooms'] <= 12, X['bathrooms'], training_data_column_median.T['bathrooms'])
    X['bathrooms'] = X['bathrooms'].astype('float32')

    X['sqft_living'] = np.where(X['sqft_living'] > 0, X['sqft_living'], training_data_column_median.T['sqft_living'])
    X['sqft_living'] = X['sqft_living'].astype('uint32')

    if not is_test:
        return X, y
    return X


def preprocess_categorical(X, is_test):
    """
    Preprocess the categorical variables in the given dataset by handling zipcode and year built features.
    If the input dataset is a test set, the function uses the columns from the training set to ensure consistency.

    Parameters:
        X (pd.DataFrame): The input dataset containing the categorical features to be preprocessed.
        is_test (bool): A flag indicating whether the input dataset is a test set or not.

    Returns:
        X (pd.DataFrame): The preprocessed dataset with the categorical features encoded as dummy variables.

    """
    global train_columns
    for col_name in CATEGORICAL_COLUMNS:
        X[f"{col_name}_other"] = 0
        X = pd.get_dummies(X, columns=[col_name], prefix=f"{col_name}", dtype=np.dtype('uint8'))
    if is_test:
        X = X.reindex(columns=train_columns, fill_value=0)
    else:
        train_columns = X.columns
    return X


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
    # Iterate over each feature and create a scatter plot with the response variable
    for col in X.columns:
        # Calculate Pearson correlation between feature and response
        # Unbiased std estimators
        std_x = np.std(X[col], ddof=1)
        std_y = np.std(y, ddof=1)
        if std_x != 0 and std_y != 0:
            pearson_correlation = np.cov(X[col].values, y.values, ddof=1)[0, 1] / (std_x * std_y)
        else:
            pearson_correlation = 0

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
    """
    Generate a plotly figure displaying the average loss as a function of training size.
    This function creates a plotly figure with three lines, representing the average loss, and the lower and upper
    bounds of the confidence interval, respectively. The confidence interval is shaded in light grey.

    Parameters:
        p_values (np.ndarray): A list of percentages corresponding to the training set size.
        loss_values (np.ndarray): A list of average loss values for each percentage in p_values.
        std_loss_values (np.ndarray): A list of standard deviation values for the loss corresponding to each percentage in
        p_values.

    Returns:
        go.Figure: The function displays the plotly figure.
    """
    return go.Figure([go.Scatter(x=p_values, y=loss_values,
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


def fit_and_predict_model(p_values, loss_values, std_loss_values):
    """
    Fit a linear regression model on a dataset and predict its loss for different percentages of sampled data.
    This function iterates through the given percentages (p_values) and samples the dataset accordingly. For each
    percentage, the linear regression model is fitted ITERATIONS number of times on the sampled data, and the mean loss
    and standard deviation of the loss are calculated for each percentage.

    Parameters:
        p_values (np.ndarray): A list of percentages to sample the dataset for training.
        loss_values (np.ndarray): A list to store the mean loss for each percentage of sampled data.
        std_loss_values (np.ndarray): A list to store the standard deviation of the loss for each percentage of sampled data.

    Returns:
        tuple: A tuple containing the updated lists of mean loss (loss_values) and standard deviation of the loss
        (std_loss_values).
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
    df = remove_na_prices(df)
    train_x, train_y, test_x, test_y = split_train_test(df.drop('price', axis=1), df['price'])
    print("Data split into train and test")

    # Question 2 - Preprocessing of housing prices dataset
    processed_train_x, processed_train_y = preprocess_data(train_x, train_y)
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
    fig = generate_regression_figure(training_percentage, loss_avg_per_batch, std_loss_avg_per_batch)
    fig.show()
    print("Finished Training")
