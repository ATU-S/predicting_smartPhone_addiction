from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from .config import TARGET_COL, TEST_SIZE, RANDOM_STATE, MIN_SAMPLES_FOR_STRATIFY
from .logger import get_logger

logger = get_logger("Preprocessing")

def encode_target(df):
    if df[TARGET_COL].dtype == "object":
        encoder = LabelEncoder()
        df[TARGET_COL] = encoder.fit_transform(df[TARGET_COL])
        logger.info("Target column encoded")
    return df

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
