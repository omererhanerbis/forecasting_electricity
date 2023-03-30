"""Tasks running the core analyses."""

import pandas as pd
import pytask

from forecasti_electricity.analysis.model import fit_logit_model, load_model
from forecasti_electricity.analysis.predict import predict_prob_by_age
from forecasti_electricity.config import BLD, GROUPS, SRC
from forecasti_electricity.utilities import read_yaml


