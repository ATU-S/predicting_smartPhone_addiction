import pandas as pd
import numpy as np
from .data_loader import load_data
from .preprocessing import encode_target, split_and_scale
from .feature_sets import get_feature_sets
from .models import get_models
from .evaluation import evaluate
from .logger import get_logger
from .timeseries_analyzer import detect_escalation_patterns, identify_critical_features
from .risk_profiler import batch_profile_users, PERSONA_DEFINITIONS
from .schema import BEHAVIORAL_FEATURES, PSYCHOLOGICAL_FEATURES

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
    
    # Profile users by persona
    best_model = models["Random Forest"]
    best_model.fit(feature_sets["Combined"], y)
    predictions = best_model.predict_proba(feature_sets["Combined"])[:, 1]
    personas = batch_profile_users(feature_sets["Combined"], predictions)
    
    # Log insights
    logger.info("Escalation Risk Summary: %s", escalation_summary)
    logger.info("Top Critical Features: %s", list(critical_features.keys())[:3])
    logger.info("Persona Distribution:\n%s", personas["persona"].value_counts())
    
    return results_df
