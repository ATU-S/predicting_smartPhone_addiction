from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from .config import TARGET_COL, TEST_SIZE, RANDOM_STATE

def encode_target(df):
    """
    Encodes the target column if it is categorical.
    """
    if df[TARGET_COL].dtype == "object":
        encoder = LabelEncoder()
        df[TARGET_COL] = encoder.fit_transform(df[TARGET_COL])
    return df

def split_and_scale(X, y):
    """
    Splits data and applies standard scaling.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
