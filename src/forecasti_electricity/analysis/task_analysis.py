"""Tasks running the core analyses."""

import re

import pandas as pd
import pmdarima as pm
import pytask
from statsmodels.tsa.arima.model import ARIMA

from forecasti_electricity.analysis.model import (
    consumption_outlier_smoother,
    critical_thresholds,
    decompose_time_series_data,
    dependent_variable_data_reducer,
    load_model,
    outlier_finder,
    weekday_mean_calculator,
)
from forecasti_electricity.analysis.predict import (
    daily_naive_predictions,
)
from forecasti_electricity.config import BLD, SRC
from forecasti_electricity.utilities import read_yaml


@pytask.mark.depends_on(
    {
        "scripts": ["model.py", "predict.py"],
        "data": BLD / "python" / "data" / "train_dependent.csv",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.produces(BLD / "python" / "models" / "naive_arima_model.pickle")
def task_fit_naive_arima_model(depends_on, produces):
    """Fit a logistic regression model (Python version)."""
    read_yaml(depends_on["data_info"])
    data = pd.read_csv(depends_on["data"])
    decomposition = decompose_time_series_data(data)
    critical_values = critical_thresholds(decomposition)
    outliers = outlier_finder(data, decomposition, critical_values)
    weekday_means = weekday_mean_calculator(data)
    smoothed_data = consumption_outlier_smoother(data, outliers, weekday_means)
    dependent_variable = dependent_variable_data_reducer(
        smoothed_data,
        ["date", "weekday"],
    )

    grid_model = pm.auto_arima(y=dependent_variable, seasonal=True, m=7)
    grid_model_string = str(grid_model)
    grid_model_values = [int(s) for s in re.findall(r"-?\d+\.?\d*", grid_model_string)]
    model = ARIMA(
        dependent_variable,
        order=(grid_model_values[0:3]),
        seasonal_order=grid_model_values[3:7],
    )
    model_fit = model.fit()
    model_fit.save(produces)


@pytask.mark.depends_on(
    {
        "scripts": ["model.py", "predict.py"],
        "data": BLD / "python" / "data" / "train_dependent.csv",
        "model": BLD / "python" / "models" / "naive_arima_model.pickle",
    },
)
@pytask.mark.produces(BLD / "python" / "predictions" / "naive_arima_predictions.csv")
def task_predict(depends_on, produces):
    """Forecasting the model estimates."""
    model = load_model(depends_on["model"])
    predicted_prob = daily_naive_predictions(model)
    predicted_prob.to_csv(produces, index=False)
