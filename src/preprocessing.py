from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .config import TARGET_COL, TEST_SIZE, RANDOM_STATE, MIN_SAMPLES_FOR_STRATIFY, TARGET_MAPPING
from .logger import get_logger

logger = get_logger("Preprocessing")

def encode_target(df):
    if df[TARGET_COL].dtype == "object":
        unknown = set(df[TARGET_COL].unique()) - set(TARGET_MAPPING.keys())
        if unknown:
            raise ValueError(f"Unknown target labels: {sorted(unknown)}")
        df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAPPING)
        logger.info("Target column encoded using fixed label mapping")
    return df

def split_data(X, y):
    stratify = y if len(y) >= MIN_SAMPLES_FOR_STRATIFY else None

    if stratify is None:
        logger.warning("Stratification disabled due to small dataset")

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify
    )

def split_and_scale(X, y):
    stratify = y if len(y) >= MIN_SAMPLES_FOR_STRATIFY else None

    if stratify is None:
        logger.warning("Stratification disabled due to small dataset")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
