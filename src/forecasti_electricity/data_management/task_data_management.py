"""Tasks for managing the data."""

import pandas as pd
import pytask
import requests
import numpy as np
import datetime as dt


from forecasti_electricity.config import BLD, SRC
from forecasti_electricity.data_management.restructure_data import (
    daily_conversion,
    clean_data,
    manipulate_data,
    weekday_feature_adder,
)
from forecasti_electricity.utilities import read_yaml


@pytask.mark.depends_on(
    {
        "scripts": ["restructure_data.py"],
        "data_info": SRC / "data_management" / "data_info.yaml",
        "electricity_data": SRC / "data" / "RealTimeConsumption-01012017-28022023.csv",
    },
)
@pytask.mark.produces(
    {
        "hourly": BLD / "python" / "data" / "hourly_data_clean.csv",
        "daily": BLD / "python" / "data" / "daily_data_clean.csv",
    },
)
def task_operatetable_data_python(depends_on, produces):
    """Conform the data to the structure to perform python operations."""
    data_info = read_yaml(depends_on["e_data_info"])
    data = pd.read_csv(depends_on["electricity_data"], skiprows=[1])
    data = clean_data(data, data_info)
    data = manipulate_data(data)
    daily_data = daily_conversion(data)
    daily_data = weekday_feature_adder(daily_data)
    daily_data.to_csv(produces["daily"], index=True)