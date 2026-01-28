from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "smartphone_addiction.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "feature_comparison_results.csv"

TARGET_COL = "Addiction_Level"

TEST_SIZE = 0.2
RANDOM_STATE = 42

MIN_SAMPLES_FOR_STRATIFY = 30
