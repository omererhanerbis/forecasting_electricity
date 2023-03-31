"""Performance measure functions."""


def perf_dt(actual, forecast):
    """Calculate several performance measures.

    Args:
        actual (pandas.DataFrame): The actual values data set.
        forecast (pandas.DataFrame): The forecast values data set.

    Returns:
        dictionary: The dictionary containing performance measure names and values

    """
    n = len(actual)
    error = actual - forecast
    my_mean = actual.mean()
    my_sd = actual.std()
    FBias = sum(error) / sum(actual)
    sum(error / actual) / n
    MAPE = sum(abs(error / actual)) / n
    RMSE = (sum(error * error) ** (1 / 2)) / n
    MAD = sum(abs(error)) / n
    WMAPE = MAD / my_mean
    l = {
        "n": n,
        "mean": my_mean,
        "std": my_sd,
        "Fbias": FBias,
        "MAPE": MAPE,
        "RMSE": RMSE,
        "MAD": MAD,
        "WMAPE": WMAPE,
    }
    return l
