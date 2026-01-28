import pandas as pd
from .data_loader import load_data
from .preprocessing import encode_target, split_and_scale
from .feature_sets import get_feature_sets
from .models import get_models
from .evaluation import evaluate
from .logger import get_logger

logger = get_logger("Experiment")

def run_experiment():
    df = encode_target(load_data())
    feature_sets, y = get_feature_sets(df)
    models = get_models()

    records = []

    for model_name, model in models.items():
        for feature_name, X in feature_sets.items():
            X_train, X_test, y_train, y_test = split_and_scale(X, y)
            metrics = evaluate(model, X_train, X_test, y_train, y_test)

            record = {
                "Model": model_name,
                "Feature Set": feature_name,
                **metrics
            }
            records.append(record)

            logger.info(
                "%s | %s | Acc: %.3f | F1: %.3f",
                model_name,
                feature_name,
                metrics["Accuracy"],
                metrics["F1-score"]
            )

    return pd.DataFrame(records)
