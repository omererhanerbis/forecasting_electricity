"""Tasks for managing the data."""


import pandas as pd
import pytask

from forecasti_electricity.config import BLD, SRC
from forecasti_electricity.data_management.create_features import (
    sunlight_data,
)
from forecasti_electricity.data_management.restructure_data import (
    clean_data,
    daily_conversion,
    manipulate_data,
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
    data_info = read_yaml(depends_on["data_info"])
    data = pd.read_csv(depends_on["electricity_data"], skiprows=[1])
    data = clean_data(data, data_info)
    data = manipulate_data(data)
    hourly_data = data

    hourly_data.to_csv(produces["hourly"], index=False)
    daily_data = daily_conversion(data)
    daily_data.to_csv(produces["daily"], index=True)


@pytask.mark.depends_on(
    {
        "scripts": ["restructure_data.py"],
        "data_info": SRC / "data_management" / "data_info.yaml",
        "currency": SRC / "data" / "USDTRY=X.csv",
    },
)
@pytask.mark.produces(BLD / "python" / "data" / "currency_data.csv")
def task_currency_data(depends_on, produces):
    """Obtain and clear currency data."""
    read_yaml(depends_on["data_info"])
    data = pd.read_csv(depends_on["currency"], skiprows=[1])
    data = data.drop(["Open", "High", "Low", "Adj Close", "Volume"], axis=1)
    data = data.rename(columns={"Date": "date", "Close": "currency"})
    data.to_csv(produces, index=False)


@pytask.mark.depends_on(
    {
        "scripts": ["restructure_data.py"],
        "data_info": SRC / "data_management" / "data_info.yaml",
        "sunlight": BLD / "python" / "data" / "sunlight_data.csv",
        "currency": BLD / "python" / "data" / "currency_data.csv",
    },
)
@pytask.mark.produces(BLD / "python" / "data" / "all_features_data.csv")
def task_all_features_data(depends_on, produces):
    read_yaml(depends_on["data_info"])
    sunlight = pd.read_csv(depends_on["sunlight"])
    currency = pd.read_csv(depends_on["currency"])

    all_features = pd.merge(sunlight, currency, how="left", on="date")
    all_features = all_features.ffill()
    all_features.to_csv(produces, index=False)


@pytask.mark.depends_on(
    {
        "scripts": ["restructure_data.py"],
        "data_info": SRC / "data_management" / "data_info.yaml",
        "features": BLD / "python" / "data" / "all_features_data.csv",
        "dependent": BLD / "python" / "data" / "daily_data_clean.csv",
    },
)
@pytask.mark.produces(
    {
        "train_features": BLD / "python" / "data" / "train_features_data.csv",
        "test_features": BLD / "python" / "data" / "test_features_data.csv",
        "train_dependent": BLD / "python" / "data" / "train_dependent.csv",
        "test_dependent": BLD / "python" / "data" / "test_dependent.csv",
    },
)
def task_train_test_divider(depends_on, produces):
    """Divide data into train and test sets."""
    read_yaml(depends_on["data_info"])
    features = pd.read_csv(depends_on["features"])
    dependent = pd.read_csv(depends_on["dependent"])

    train_features = features.loc[
        (features["date"] <= "2023-02-21") & (features["date"] >= "2017-01-03")
    ]
    test_features = features[
        (features["date"] > "2023-02-21") & (features["date"] <= "2023-02-28")
    ]
    train_dependent = dependent[
        (dependent["date"] <= "2023-02-21") & (dependent["date"] >= "2017-01-03")
    ]
    test_dependent = dependent[
        (dependent["date"] > "2023-02-21") & (dependent["date"] <= "2023-02-28")
    ]

    train_features.to_csv(produces["train_features"], index=False)
    test_features.to_csv(produces["test_features"], index=False)
    train_dependent.to_csv(produces["train_dependent"], index=False)
    test_dependent.to_csv(produces["test_dependent"], index=False)


@pytask.mark.produces(BLD / "python" / "data" / "sunlight_data.csv")
def task_sunlight_data(produces):
    """Request and calculate sunlight time from prayer times API."""
    sunlight = sunlight_data()
    sunlight = sunlight.drop(["sunrise", "sunset"], axis=1)
    sunlight.to_csv(produces, index=False)
