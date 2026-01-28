import pandas as pd
from .config import DATA_PATH

def load_data():
    """
    Loads the smartphone addiction dataset.
    """
    df = pd.read_csv(DATA_PATH)
    return df
