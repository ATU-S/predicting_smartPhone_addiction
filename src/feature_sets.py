from .schema import BEHAVIORAL_FEATURES, PSYCHOLOGICAL_FEATURES
from .config import TARGET_COL

def get_feature_sets(df):
    X_behavioral = df[BEHAVIORAL_FEATURES]
    X_psychological = df[PSYCHOLOGICAL_FEATURES]
    X_combined = df[BEHAVIORAL_FEATURES + PSYCHOLOGICAL_FEATURES]
    y = df[TARGET_COL]

    return {
        "Behavioral": X_behavioral,
        "Psychological": X_psychological,
        "Combined": X_combined
    }, y
