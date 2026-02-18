from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "smartphone_addiction.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "feature_comparison_results.csv"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
BEST_MODEL_FILE = ARTIFACTS_DIR / "best_model.joblib"
MODEL_META_FILE = ARTIFACTS_DIR / "best_model_meta.json"
CONFUSION_MATRIX_PNG = RESULTS_DIR / "confusion_matrix.png"
ROC_CURVE_PNG = RESULTS_DIR / "roc_curve.png"
METRICS_SUMMARY_FILE = RESULTS_DIR / "metrics_summary.json"
FEATURE_IMPORTANCE_FILE = RESULTS_DIR / "feature_importance.csv"

TARGET_COL = "Addiction_Level"
TARGET_LABELS = ["Low", "Moderate", "High"]
TARGET_MAPPING = {label: idx for idx, label in enumerate(TARGET_LABELS)}
HIGH_RISK_LABEL = "High"
SELECTION_METRIC = "F1-weighted"

TEST_SIZE = 0.2
RANDOM_STATE = 42

MIN_SAMPLES_FOR_STRATIFY = 30
