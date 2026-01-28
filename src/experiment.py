import pandas as pd
from .data_loader import load_data
from .preprocessing import encode_target, split_and_scale
from .feature_sets import get_feature_sets
from .models import get_models
from .evaluation import evaluate

def run_experiment():
    """
    Runs behavioral vs psychological vs combined feature analysis
    using multiple ML models.
    """
    df = load_data()
    df = encode_target(df)

    Xb, Xp, Xc, y = get_feature_sets(df)
    models = get_models()

    results = []

    for model_name, model in models.items():
        for feature_label, X in zip(
            ["Behavioral", "Psychological", "Combined"],
            [Xb, Xp, Xc]
        ):
            X_train, X_test, y_train, y_test = split_and_scale(X, y)
            acc, f1 = evaluate(model, X_train, X_test, y_train, y_test)

            results.append([
                model_name,
                feature_label,
                acc,
                f1
            ])

    results_df = pd.DataFrame(
        results,
        columns=["Model", "Feature Set", "Accuracy", "F1-score"]
    )

    return results_df
