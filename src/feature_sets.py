from .config import TARGET_COL

# ------------------------------
# Feature Group Definitions
# ------------------------------

BEHAVIORAL_FEATURES = [
    "screen_time",
    "phone_unlocks",
    "night_usage",
    "social_media_hours",
    "gaming_hours",
    "notifications"
]

PSYCHOLOGICAL_FEATURES = [
    "stress_level",
    "anxiety_score",
    "sleep_quality",
    "self_control",
    "loneliness"
]

def get_feature_sets(df):
    """
    Returns behavioral, psychological, combined feature sets and target.
    """
    X_behavioral = df[BEHAVIORAL_FEATURES]
    X_psychological = df[PSYCHOLOGICAL_FEATURES]
    X_combined = df[BEHAVIORAL_FEATURES + PSYCHOLOGICAL_FEATURES]
    y = df[TARGET_COL]

    return X_behavioral, X_psychological, X_combined, y
