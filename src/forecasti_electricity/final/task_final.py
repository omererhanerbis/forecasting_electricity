"""Tasks running the results formatting (tables, figures)."""


import pandas as pd
import pytask

from forecasti_electricity.analysis.model import load_model
from forecasti_electricity.config import BLD, SRC
from forecasti_electricity.final.performance_measure import perf_dt
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


@pytask.mark.depends_on(BLD / "python" / "models" / "naive_arima_model.pickle")
@pytask.mark.produces(BLD / "python" / "tables" / "naive_arima_model.tex")
def task_create_naive_results_table_python(depends_on, produces):
    """Store a table in LaTeX format with the estimation results (Python version)."""
    model = load_model(depends_on)
    table = model.summary().as_latex()
    with open(produces, "w") as f:
        f.writelines(table)


@pytask.mark.depends_on(BLD / "python" / "models" / "featured_arima_model.pickle")
@pytask.mark.produces(BLD / "python" / "tables" / "featured_arima_model.tex")
def task_create_featured_results_table_python(depends_on, produces):
    """Store a table in LaTeX format with the estimation results (Python version)."""
    model = load_model(depends_on)
    table = model.summary().as_latex()
    with open(produces, "w") as f:
        f.writelines(table)


@pytask.mark.depends_on(
    {
        "actual": BLD / "python" / "data" / "test_dependent.csv",
        "forecast": BLD / "python" / "predictions" / "naive_arima_predictions.csv",
    },
)
@pytask.mark.produces(BLD / "python" / "tables" / "naive_arima_performance.txt")
def task_naive_predictions_table_python(depends_on, produces):
    """Store a table in LaTeX format with the forecast performance."""
    actual = pd.read_csv(depends_on["actual"])
    forecast = pd.read_csv(depends_on["forecast"])

    actual_cons = actual.consumption
    forecast_cons = forecast.predicted_mean

    performance_result = perf_dt(actual=actual_cons, forecast=forecast_cons)

    with open(produces, "w") as f:
        print(performance_result, file=f)


@pytask.mark.depends_on(
    {
        "actual": BLD / "python" / "data" / "test_dependent.csv",
        "forecast": BLD / "python" / "predictions" / "featured_arima_predictions.csv",
    },
)
@pytask.mark.produces(BLD / "python" / "tables" / "featured_arima_performance.txt")
def task_featured_predictions_table_python(depends_on, produces):
    """Store a table in LaTeX format with the forecast performance."""
    actual = pd.read_csv(depends_on["actual"])
    forecast = pd.read_csv(depends_on["forecast"])

    actual_cons = actual.consumption
    forecast_cons = forecast.predicted_mean

    performance_result = perf_dt(actual=actual_cons, forecast=forecast_cons)

    with open(produces, "w") as f:
        print(performance_result, file=f)
