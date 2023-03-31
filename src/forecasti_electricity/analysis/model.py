"""Functions for fitting the regression model."""

import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.iolib.smpickle import load_pickle
from statsmodels.tsa.seasonal import seasonal_decompose

from forecasti_electricity.data_management.restructure_data import (
    weekday_feature_adder,
)


def load_model(path):
    """Load statsmodels model.

    Args:
        path (str or pathlib.Path): Path to model file.

    Returns:
        statsmodels.base.model.Results: The stored model.

    """
    return load_pickle(path)


def decompose_time_series_data(data):
    """Decompose time series data into trend seasonality and residuals.

    Args:
        data (pandas.DataFrame): The data set.

    Returns:
        statsmodels.tsa.seasonal.DecomposeResult: The decomposition of input data.

    """
    decomposed_data = seasonal_decompose(data.consumption, model="additive", period=7)

    return decomposed_data


def critical_thresholds(data, confidence_levels=[0.95]):
    """Calculate critical threshold values to find outliers of the data.

    Args:
        data (statsmodels.tsa.seasonal.DecomposeResult): The decomposed input data
        confidence_levels (list) : The condfidence levels to find critical values.

    Returns:
        list: The critical threshold values for specified confidence levels, in the order of all upper levels and all lower levels

    """
    data = data.resid
    upper_thresholds = (1 - np.array(confidence_levels)) / 2
    lower_thresholds = 1 - upper_thresholds
    thresholds = np.append(upper_thresholds, lower_thresholds)

    ci_vals = [st.norm.ppf(i) for i in thresholds]

    critical_vals = 0 + data.std() * np.array(ci_vals)
    critical_vals = critical_vals.tolist()

    return critical_vals


def outlier_finder(core_data, decomposed_data, critical_values):
    """Locate outlier data points in core data.

    Args:
        core_data (pandas.DataFrame): The main data set.
        decomposed_data (pandas.DataFrame): The decomposed data that contains residuals which outlier decider works on.
        critical_values (list): The threshold values to decide the outlier bounds.

    Returns:
        pandas.DataFrame: The outlier core data points

    """
    data = core_data
    residuals = decomposed_data.resid
    thresholds = critical_values

    outliers = pd.DataFrame(
        data[(residuals < thresholds[0]) | (residuals > thresholds[1])],
    )

    return outliers


def weekday_mean_calculator(data):
    """Calculate weekday means of data.

    Args:
        data (pandas.DataFrame): The data set.

    Returns:
        pandas.DataFrame: The weekday means of data

    """
    data = weekday_feature_adder(data)
    day_means = data.groupby("weekday").mean(numeric_only=True)

    return day_means


def consumption_outlier_smoother(data, outliers_data, smoothing_data):
    """Smooth data at hand with given smoother values.

    Args:
        data (pandas.DataFrame): The data set.
        outliers_data (pandas.core.frame.DataFrame): outliers data set.
        smoothing_data (pandas.DataFrame): The smoothing data set.

    Returns:
        pandas.DataFrame: The smoothed data.

    """
    data = weekday_feature_adder(data)
    for i in outliers_data.index:
        data.loc[i, "consumption"] = smoothing_data.loc[
            data.loc[i, "weekday"],
            "consumption",
        ]

    return data


def dependent_variable_data_reducer(data, drop_list):
    """Reduce data to dependent variable only with index.

    Args:
        data (pandas.DataFrame): The data set.
        drop_list (list): the list of column names to drop

    Returns:
        pandas.DataFrame: The reduced dependent variable.

    """
    data = data.drop(drop_list, axis=1)

    return data
