"""Tasks running the results formatting (tables, figures)."""

import pandas as pd
import pytask

from forecasti_electricity.analysis.model import load_model
from forecasti_electricity.config import BLD, GROUPS, SRC
from forecasti_electricity.final import plot_regression_by_age
from forecasti_electricity.utilities import read_yaml
