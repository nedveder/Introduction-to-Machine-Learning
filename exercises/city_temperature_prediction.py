import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"
DEGREE_TO_ITERATE = 10
SELECTED_DEGREE = 6


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    # Fix labels
    df['Temp'] = df.groupby(['Month', 'City'])['Temp'].transform(lambda x: x.mask(x.lt(-70), x.mean()))
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df


def israel_data_exploration(sample_df):
    fig1 = px.scatter(sample_df, x='DayOfYear', y='Temp', color='Year'
                      , color_discrete_sequence=px.colors.qualitative.Set3)
    fig1.update_layout(title='Temperature Variation by Day of Year and Year', xaxis_title='Day of Year',
                       yaxis_title='Temperature (°C)')
    fig1.show()

    # Grouping the samples by month and calculating the standard deviation of daily temperatures
    monthly_std = sample_df.groupby('Month')['Temp'].agg('std')
    fig2 = px.bar(monthly_std, x=monthly_std.index, y=monthly_std.values, color=monthly_std.values)
    fig2.update_layout(title='Temperature Variation by Month', xaxis_title='Month',
                       yaxis_title='Temperature Standard Deviation (°C)')
    fig2.show()


def data_exploration(sample_df):
    avg_temp = sample_df.groupby(['Country', 'Month']).agg({'Temp': ['mean', 'std']})
    avg_temp.columns = ['avg_temp', 'std_temp']
    avg_temp.reset_index(inplace=True)

    # Creating a line plot with error bars
    fig = px.line(avg_temp, x='Month', y='avg_temp', color='Country', error_y='std_temp',
                  title='Average Monthly Temperature by Country')
    fig.update_layout(xaxis_title='Month', yaxis_title='Temperature (°C)')

    # Showing the plot
    fig.show()


def fit_model_over_israel(israel_df):
    # Splitting the dataset into a training set (75%) and test set (25%)
    train_x, train_y, test_x, test_y = split_train_test(israel_df.drop('Temp', axis=1), israel_df['Temp'])

    test_errors = np.zeros(DEGREE_TO_ITERATE)

    # Looping over k values
    for k in range(1, DEGREE_TO_ITERATE + 1):
        # New model with degree k
        polynomial_fitting = PolynomialFitting(k)
        polynomial_fitting.fit(train_x['DayOfYear'].to_numpy(dtype='float64'), train_y.to_numpy(dtype='float64'))

        # Calculating the test error
        loss = polynomial_fitting.loss(test_x['DayOfYear'].to_numpy(dtype='float64'), test_y.to_numpy(dtype='float64'))
        test_errors[k - 1] = loss

    # Printing the test errors for each value of k
    for k, error in enumerate(test_errors):
        print(f"k={k + 1}: {error:.2f}")

    # Creating a bar plot of the test errors
    fig = px.bar(x=[f"{k}" for k in range(1, DEGREE_TO_ITERATE + 1)], y=test_errors, color=test_errors)
    fig.update_layout(title='Test Errors for Polynomial Models of Different Degrees', xaxis_title='Degree',
                      yaxis_title='Test Error (MSE)', uniformtext_minsize=8, uniformtext_mode='hide')
    fig.show()


def fit_model_over_all_countries(data):
    polynomial_fitting = PolynomialFitting(SELECTED_DEGREE)
    israel_df = df[df['Country'] == 'Israel']
    train_x, train_y = israel_df.drop('Temp', axis=1)['DayOfYear'], israel_df['Temp']
    polynomial_fitting.fit(train_x.to_numpy(dtype='float64'),
                           train_y.to_numpy(dtype='float64'))

    loss_per_country = np.zeros(df['Country'].unique().shape[0])
    for i, country in enumerate(df['Country'].unique()):
        country_df = df[df['Country'] == country]
        test_x, test_y = country_df.drop('Temp', axis=1)['DayOfYear'], country_df['Temp']
        loss_per_country[i] = polynomial_fitting.loss(test_x.to_numpy(dtype='float64'),
                                                      test_y.to_numpy(dtype='float64'))

    # Creating a bar plot of the errors
    fig = px.bar(df, x=df['Country'].unique(), y=loss_per_country, color=loss_per_country)
    fig.update_layout(title='Test Errors of Model Trained on Israel Data on Other Countries', xaxis_title='Country',
                      yaxis_title='Test Error (MSE)', uniformtext_minsize=8, uniformtext_mode='hide')
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    # israel_data_exploration(df[df['Country'] == 'Israel'])

    # Question 3 - Exploring differences between countries
    # data_exploration(df)

    # Question 4 - Fitting model for different values of `k`
    # fit_model_over_israel(df[df['Country'] == 'Israel'])

    # Question 5 - Evaluating fitted model on different countries
    fit_model_over_all_countries(df)
