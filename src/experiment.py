import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from .data_loader import load_data
from .preprocessing import encode_target, split_data
from .feature_sets import get_feature_sets
from .models import get_models
from .evaluation import evaluate, save_confusion_matrix, save_roc_curve, save_metrics_summary
from .logger import get_logger
from .timeseries_analyzer import detect_escalation_patterns, identify_critical_features
from .risk_profiler import batch_profile_users, PERSONA_DEFINITIONS
from .schema import BEHAVIORAL_FEATURES, PSYCHOLOGICAL_FEATURES
from .config import (
    RESULTS_DIR,
    ARTIFACTS_DIR,
    BEST_MODEL_FILE,
    MODEL_META_FILE,
    CONFUSION_MATRIX_PNG,
    ROC_CURVE_PNG,
    METRICS_SUMMARY_FILE,
    FEATURE_IMPORTANCE_FILE,
    TARGET_LABELS,
    SELECTION_METRIC
)

logger = get_logger("Experiment")

def run_experiment():
    df = encode_target(load_data())
    feature_sets, y = get_feature_sets(df)
    models = get_models()

    records = []
    best_score = -1
    best_bundle = None

    for model_name, model in models.items():
        for feature_name, X in feature_sets.items():
            X_train, X_test, y_train, y_test = split_data(X, y)
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])
            metrics, y_pred, y_proba = evaluate(pipeline, X_train, X_test, y_train, y_test)

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
                metrics["F1-weighted"]
            )

            score = metrics.get(SELECTION_METRIC)
            if score is not None and score > best_score:
                best_score = score
                best_bundle = {
                    "model_name": model_name,
                    "feature_name": feature_name,
                    "pipeline": pipeline,
                    "metrics": metrics,
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                    "y_pred": y_pred,
                    "y_proba": y_proba,
                    "feature_columns": list(X.columns)
                }

    results_df = pd.DataFrame(records)
    
    # Add persona and time-series insights
    logger.info("Analyzing escalation patterns and risk personas...")
    
    # Detect escalation patterns across behavioral features
    escalation = detect_escalation_patterns(df, BEHAVIORAL_FEATURES)
    escalation_summary = {
        col: metrics["risk_level"] 
        for col, metrics in escalation.items()
    }
    
    # Identify critical features driving addiction
    critical_features = identify_critical_features(
        df, BEHAVIORAL_FEATURES, PSYCHOLOGICAL_FEATURES
    )
    
    # Profile users by persona using the best model
    if best_bundle is not None:
        best_pipeline = best_bundle["pipeline"]
        best_features = feature_sets[best_bundle["feature_name"]]
        best_pipeline.fit(best_features, y)
        predictions = best_pipeline.predict_proba(best_features)[:, 1]
    else:
        predictions = np.random.rand(len(df))
    personas = batch_profile_users(feature_sets["Combined"], predictions)
    
    # Log insights
    logger.info("Escalation Risk Summary: %s", escalation_summary)
    logger.info("Top Critical Features: %s", list(critical_features.keys())[:3])
    logger.info("Persona Distribution:\n%s", personas["persona"].value_counts())

    # Save best model artifacts and evaluation visuals
    if best_bundle is not None:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        best_pipeline = best_bundle["pipeline"]

        # Save confusion matrix and ROC curve from the best model
        save_confusion_matrix(
            best_bundle["y_test"],
            best_bundle["y_pred"],
            TARGET_LABELS,
            CONFUSION_MATRIX_PNG
        )
        save_roc_curve(
            best_bundle["y_test"],
            best_bundle["y_proba"],
            TARGET_LABELS,
            ROC_CURVE_PNG
        )

        # Save permutation feature importance
        perm = permutation_importance(
            best_pipeline,
            best_bundle["X_test"],
            best_bundle["y_test"],
            n_repeats=25,
            random_state=42,
            n_jobs=-1
        )
        importance_df = pd.DataFrame({
            "feature": best_bundle["feature_columns"],
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std
        }).sort_values("importance_mean", ascending=False)
        importance_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

        # Fit on full data for deployment
        best_pipeline.fit(feature_sets[best_bundle["feature_name"]], y)
        try:
            import joblib
            joblib.dump(best_pipeline, BEST_MODEL_FILE)
        except Exception as exc:
            logger.warning("Failed to persist model: %s", exc)

        meta = {
            "model_name": best_bundle["model_name"],
            "feature_set": best_bundle["feature_name"],
            "selection_metric": SELECTION_METRIC,
            "best_score": best_score,
            "metrics": best_bundle["metrics"],
            "labels": TARGET_LABELS,
            "trained_at": datetime.utcnow().isoformat() + "Z"
        }
        MODEL_META_FILE.write_text(json.dumps(meta, indent=2))

        label_counts = df["Addiction_Level"].value_counts().sort_index()
        label_distribution = {
            TARGET_LABELS[int(idx)]: int(count)
            for idx, count in label_counts.items()
        }
        summary = {
            "dataset_size": int(len(df)),
            "label_distribution": label_distribution,
            "best_model": meta
        }
        save_metrics_summary(summary, METRICS_SUMMARY_FILE)
    
    return results_df
