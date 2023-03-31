"""Functions for predicting outcomes based on the estimated model."""

import pandas as pd


def daily_predictions(model, exogen=None, forecast_window=7):
    """Forecasting daily electricity consumption from ARIMA models.

    Args:
        model (statsmodels.base.model.Results): The fitted model.
        forecast_window (int): number of days to forecast
        exog (pandas.DataFrame): Test features to be used in forecasting, default = None

    Returns:
        pandas.DataFrame: Forecasts

    """
    forecast_test = model.forecast(forecast_window, exog=exogen)

    forecast = pd.DataFrame(forecast_test)
    return forecast
