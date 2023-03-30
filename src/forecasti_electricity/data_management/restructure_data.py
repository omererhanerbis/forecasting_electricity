"""Function(s) for cleaning, restructuring and manipulating the data set(s)."""

import datetime as dt
import pandas as pd
import requests

def clean_data(data, data_info):
    """Clean data set.

    Information on data columns is stored in ``data_management/data_info.yaml``.

    Args:
        data (pandas.DataFrame): The data set.
        data_info (dict): Information on data set stored in data_info.yaml. The
            following keys can be accessed:
            - 'column_rename_mapping': Old and new names of columns to be renamend,
                stored in a dictionary with design: {'old_name': 'new_name'}
            - 'url': URL to data set

    Returns:
        pandas.DataFrame: The cleaned data set.

    """
    data = data.dropna()
    data = data.rename(columns=data_info["column_rename_mapping"])



    return data

def manipulate_data(data):
    """Manipulate the data and adjust the format of date and inputs for operations.

    Args:
        data (pandas.DataFrame): The data set.

    Returns:
        pandas.DataFrame: The manipulated data set.

    """
    data.date = data.date.apply(lambda x: dt.datetime.strptime(x, "%d.%m.%Y"), 1)
    data.consumption = data.consumption.apply(lambda x: x.replace(".", "").replace(",", ".")).astype(float)



    return data

def daily_conversion(data):
    """Convert hourly bulk data to daily data.


    Args:
        data (pandas.DataFrame): The data set.

    Returns:
        pandas.DataFrame: The restructured data set.

    """
    data = pd.DataFrame(data.groupby(by = "date").consumption.sum())



    return data


def weekday_feature_adder(data):
    """Add weekday feature to data.


    Args:
        data (pandas.DataFrame): The data set.

    Returns:
        pandas.DataFrame: The updated data set.

    """
    data["weekday"] = data.index.weekday



    return data