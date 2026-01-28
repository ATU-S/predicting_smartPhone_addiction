import pandas as pd
from .config import DATA_PATH
from .logger import get_logger

logger = get_logger("DataLoader")

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        logger.error("Dataset not found at %s", DATA_PATH)
        raise FileNotFoundError(f"Dataset missing: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    logger.info("Dataset loaded successfully with shape %s", df.shape)
    return df
