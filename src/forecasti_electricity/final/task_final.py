"""Tasks running the results formatting (tables, figures)."""

import pandas as pd
import pytask

from forecasti_electricity.config import BLD, SRC
from forecasti_electricity.final import plot_consumption
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
    fig = plot_consumption(data, data_info)
    fig.savefig(produces)


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
    fig = plot_consumption(data, data_info)
    fig.savefig(produces)
