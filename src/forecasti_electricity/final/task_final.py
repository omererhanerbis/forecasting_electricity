"""Tasks running the results formatting (tables, figures)."""

import pandas as pd
import pytask

from forecasti_electricity.config import BLD, SRC
from forecasti_electricity.final.plot import (
    plot_autocorrelations,
    plot_consumption,
    plot_decomposition,
    plot_outlier_analysis,
    plot_smoothed,
)
from forecasti_electricity.utilities import read_yaml


@pytask.mark.depends_on(
    {
        "data": BLD / "python" / "data" / "hourly_data_clean.csv",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.produces(BLD / "python" / "figures" / "hourly_consumption.png")
def task_plot_hourly_consumption(depends_on, produces):
    """Plot the consumption data."""
    data_info = read_yaml(depends_on["data_info"])
    data = pd.read_csv(depends_on["data"])
    fig_h = plot_consumption(data, data_info)
    fig_h.savefig(produces)


@pytask.mark.depends_on(
    {
        "data": BLD / "python" / "data" / "daily_data_clean.csv",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.produces(BLD / "python" / "figures" / "daily_consumption.png")
def task_plot_daily_consumption(depends_on, produces):
    """Plot the consumption data."""
    data_info = read_yaml(depends_on["data_info"])
    data = pd.read_csv(depends_on["data"])
    fig_d = plot_consumption(data, data_info)
    fig_d.savefig(produces)


@pytask.mark.depends_on(
    {
        "data": BLD / "python" / "data" / "daily_data_clean.csv",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.produces(BLD / "python" / "figures" / "autocorrelations.png")
def task_plot_autocorrelations(depends_on, produces):
    """Plot the autocorrelations of data."""
    data_info = read_yaml(depends_on["data_info"])
    data = pd.read_csv(depends_on["data"])
    fig = plot_autocorrelations(data, data_info)
    fig.savefig(produces)


@pytask.mark.depends_on(
    {
        "data": BLD / "python" / "data" / "daily_data_clean.csv",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.produces(BLD / "python" / "figures" / "decomposition.png")
def task_plot_decomposition(depends_on, produces):
    """Plot the decomposition results of the data."""
    data_info = read_yaml(depends_on["data_info"])
    data = pd.read_csv(depends_on["data"])
    fig = plot_decomposition(data, data_info)
    fig.savefig(produces)


@pytask.mark.depends_on(
    {
        "data": BLD / "python" / "data" / "daily_data_clean.csv",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.produces(BLD / "python" / "figures" / "outliers.png")
def task_plot_outliers(depends_on, produces):
    """Plot the determined outliers of the data."""
    data_info = read_yaml(depends_on["data_info"])
    data = pd.read_csv(depends_on["data"])
    fig = plot_outlier_analysis(data, data_info)
    fig = fig.figure
    fig.savefig(produces)


@pytask.mark.depends_on(
    {
        "data": BLD / "python" / "data" / "daily_data_clean.csv",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.produces(BLD / "python" / "figures" / "naive_handled_decomposition.png")
def task_plot_naive_handled_decomposition(depends_on, produces):
    """Plot the decomposition results of the data."""
    data_info = read_yaml(depends_on["data_info"])
    data = pd.read_csv(depends_on["data"])
    fig = plot_smoothed(data, data_info)
    fig.savefig(produces)
