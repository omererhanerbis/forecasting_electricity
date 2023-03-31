"""Functions plotting results."""


import seaborn as sns


def plot_consumption(data, data_info):
    """Plot regression results by age.

    Args:
        data (pandas.DataFrame): The data set.

    Returns:
        plotly.graph_objects.Figure: The figure.

    """
    sns.set(rc={"figure.figsize": (19, 8)})
    fig = sns.lineplot(data=data.consumption)
    fig = fig.get_figure()
    return fig
