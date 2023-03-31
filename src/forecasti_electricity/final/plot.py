"""Functions plotting results."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from forecasti_electricity.analysis.model import (
    consumption_outlier_smoother,
    critical_thresholds,
    decompose_time_series_data,
    outlier_finder,
    weekday_mean_calculator,
)


def plot_consumption(data, data_info):
    """Plot regression results by age.

    Args:
        data (pandas.DataFrame): The data set.

    Returns:
        plotly.graph_objects.Figure: The figure.

    """
    data.drop(["date"], axis=1)
    fig, axs = plt.subplots(1, 1)
    sns.set(rc={"figure.figsize": (19, 8)})
    fig_c = sns.lineplot(data=data, x=data.index, y="consumption")
    fig_c = fig_c.get_figure()
    return fig_c


def plot_autocorrelations(data, data_info):
    """Plot regression results by age.

    Args:
        data (pandas.DataFrame): The data set.

    Returns:
        plotly.graph_objects.Figure: The figure.

    """
    sns.set(rc={"figure.figsize": (19, 8)})

    fig, axs = plt.subplots(nrows=2)
    fig.suptitle("Correlation Graphs", y=0.96)
    xs = sm.graphics.tsa.plot_acf(data.consumption, lags=50, ax=axs[0], auto_ylims=True)
    xs = sm.graphics.tsa.plot_pacf(
        data.consumption,
        lags=50,
        ax=axs[1],
        method="ywm",
        auto_ylims=True,
    )

    return xs


def plot_decomposition(data, data_info):
    """Plot regression results by age.

    Args:
        data (pandas.DataFrame): The data set.

    Returns:
        plotly.graph_objects.Figure: The figure.

    """
    decomposed = decompose_time_series_data(data)
    # Set the figure size:
    sns.set(rc={"figure.figsize": (19, 19)})

    # Divide the figure:
    fig, axs = plt.subplots(4, 1)
    fig.suptitle("Decomposition results", y=0.92)

    # Plot each variable with frgn_deposit_share
    fig = sns.lineplot(data=decomposed.observed, ax=axs[0])
    fig = sns.lineplot(data=decomposed.trend, ax=axs[1])
    fig = sns.lineplot(data=decomposed.seasonal, ax=axs[2])
    fig = sns.lineplot(data=decomposed.resid, ax=axs[3])

    fig = fig.get_figure()
    return fig


def plot_outlier_analysis(data, data_info, confidence_levels=[0.90, 0.95, 0.99]):
    """Plot regression results by age.

    Args:
        data (pandas.DataFrame): The data set.
        confidence_levels (list): The pre-determined confidence levels of interest

    Returns:
        plotly.graph_objects.Figure: The figure.

    """
    decomposed = decompose_time_series_data(data)
    residuals = decomposed.resid
    critical_threshold_vals = critical_thresholds(decomposed, [0.90, 0.95, 0.99])

    fig, axs = plt.subplots(1, 1)
    fig = sns.scatterplot(data=residuals)
    fig = fig.get_figure()
    cols_for_criticals = (
        list(mcolors.BASE_COLORS.values())[0 : int(len(critical_threshold_vals) / 2)]
        * 2
    )
    [
        plt.axhline(
            y=critical_threshold_vals[i],
            linestyle="--",
            color=cols_for_criticals[i],
        )
        for i in list(range(0, len(critical_threshold_vals)))
    ]

    return fig


def plot_smoothed(data, data_info):
    """Plot regression results by age.

    Args:
        data (pandas.DataFrame): The data set.

    Returns:
        plotly.graph_objects.Figure: The figure.

    """
    initial_decomposed = decompose_time_series_data(data)
    critical_threshold_vals = critical_thresholds(initial_decomposed)
    outliers = outlier_finder(data, initial_decomposed, critical_threshold_vals)
    weekday_means = weekday_mean_calculator(data)
    smoothed_data = consumption_outlier_smoother(data, outliers, weekday_means)

    fig = plot_decomposition(smoothed_data, data_info=0)

    return fig
