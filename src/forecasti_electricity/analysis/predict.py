"""Functions for predicting outcomes based on the estimated model."""

import pandas as pd


def daily_predictions(model, exogen=None, forecast_window=7):
    """Predict smoking probability for varying age values.

    For each group value in column data[group] we create new data that runs through a
    grid of age values from data.age.min() to data.age.max() and fixes all column
    values to the ones returned by data.mode(), except for the group column.

    Args:
        data (pandas.DataFrame): The data set.
        model (statsmodels.base.model.Results): The fitted model.
        forecast_window (int): number of days to forecast
        exog (pandas.DataFrame): Test features to be used in forecasting, default = None

    Returns:
        pandas.DataFrame: Predictions. Has columns 'age' and one column for each
            category in column group.

    """
    forecast_test = model.forecast(forecast_window, exog=exogen)

    forecast = pd.DataFrame(forecast_test)
    return forecast
