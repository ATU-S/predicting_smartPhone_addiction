"""
Risk Profiler - Creates Personalized Risk Personas
Classifies users into behavioral archetypes for targeted interventions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .schema import BEHAVIORAL_FEATURES, PSYCHOLOGICAL_FEATURES


PERSONA_DEFINITIONS = {
    "Late-Night Gamer": {
        "description": "High night usage and gaming hours, disrupted sleep",
        "key_features": ["night_usage", "gaming_hours", "sleep_quality"],
        "thresholds": {"night_usage": 0.7, "gaming_hours": 0.6, "sleep_quality": 0.3},
        "intervention": "Establish screen-free bedtime routines; set gaming time limits"
    },
    "Social Media Addict": {
        "description": "Extreme social media consumption with high notification dependency",
        "key_features": ["social_media_hours", "notifications", "screen_time"],
        "thresholds": {"social_media_hours": 0.75, "notifications": 0.7},
        "intervention": "Implement notification controls; schedule phone-free social time"
    },
    "Stress-Driven User": {
        "description": "Uses phone as escape mechanism due to high stress/anxiety",
        "key_features": ["stress_level", "anxiety_score", "screen_time"],
        "thresholds": {"stress_level": 0.7, "anxiety_score": 0.65},
        "intervention": "Develop stress management techniques; practice mindfulness"
    },
    "Compulsive Checker": {
        "description": "Frequent phone unlocks with low self-control",
        "key_features": ["phone_unlocks", "self_control", "notifications"],
        "thresholds": {"phone_unlocks": 0.75, "self_control": 0.3},
        "intervention": "Use app blockers; practice delayed gratification techniques"
    },
    "Lonely Connector": {
        "description": "High phone usage driven by loneliness and social anxiety",
        "key_features": ["loneliness", "screen_time", "anxiety_score"],
        "thresholds": {"loneliness": 0.7, "screen_time": 0.65},
        "intervention": "Build offline social connections; join community activities"
    },
    "Balanced User": {
        "description": "Healthy relationship with phone, all metrics within normal range",
        "key_features": ["all"],
        "thresholds": {},
        "intervention": "Maintain current habits; regular digital wellness check-ins"
    }
}


def normalize_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Normalize features to 0-1 scale for consistent persona matching.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    features : list
        Feature columns to normalize
    
    Returns:
    --------
    pd.DataFrame : Normalized dataframe
    """
    df_norm = df.copy()
    
    for col in features:
        if col in df_norm.columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            
            if max_val > min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 0
    
    return df_norm


def classify_user_persona(row: pd.Series) -> Tuple[str, float]:
    """
    Classify a single user into a risk persona based on their feature profile.
    
    Parameters:
    -----------
    row : pd.Series
        User's normalized feature values
    
    Returns:
    --------
    Tuple[str, float] : (persona_name, confidence_score)
    """
    scores = {}
    
    for persona_name, definition in PERSONA_DEFINITIONS.items():
        if persona_name == "Balanced User":
            # Check if user is balanced (all features moderate)
            all_features = BEHAVIORAL_FEATURES + PSYCHOLOGICAL_FEATURES
            available_features = [f for f in all_features if f in row.index]
            mean_score = row[available_features].mean()
            
            if mean_score < 0.5:
                scores[persona_name] = 0.95
            else:
                scores[persona_name] = 0.05
        else:
            # Calculate match score for specific persona
            key_features = definition["key_features"]
            thresholds = definition["thresholds"]
            
            matches = 0
            total_checks = len(thresholds)
            
            for feature, threshold in thresholds.items():
                if feature in row.index:
                    if row[feature] >= threshold:
                        matches += 1
            
            # Confidence = proportion of thresholds met
            confidence = matches / total_checks if total_checks > 0 else 0
            scores[persona_name] = confidence
    
    # Get primary persona
    primary_persona = max(scores, key=scores.get)
    confidence = scores[primary_persona]
    
    return primary_persona, float(confidence)


def create_personalized_profile(df: pd.DataFrame, user_prediction: float) -> Dict:
    """
    Create a complete personalized risk profile for a user/cohort.
    
    Parameters:
    -----------
    df : pd.DataFrame
        User's feature data
    user_prediction : float
        Model's addiction prediction score (0-1)
    
    Returns:
    --------
    Dict : Complete personalized profile with persona, interventions, and insights
    """
    # Normalize features
    df_norm = normalize_features(df, BEHAVIORAL_FEATURES + PSYCHOLOGICAL_FEATURES)
    
    # Get first row if multiple rows provided
    if len(df_norm) > 0:
        user_row = df_norm.iloc[0]
    else:
        return {"error": "empty_dataframe"}
    
    # Classify persona
    persona_name, confidence = classify_user_persona(user_row)
    persona_def = PERSONA_DEFINITIONS[persona_name]
    
    # Identify top risk factors
    all_features = BEHAVIORAL_FEATURES + PSYCHOLOGICAL_FEATURES
    feature_scores = {f: user_row[f] for f in all_features if f in user_row.index}
    top_risk_factors = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    profile = {
        "persona": persona_name,
        "confidence": confidence,
        "addiction_risk_score": float(user_prediction),
        "description": persona_def["description"],
        "top_risk_factors": [{"factor": f, "severity": float(s)} for f, s in top_risk_factors],
        "intervention": persona_def["intervention"],
        "key_features_to_monitor": persona_def["key_features"]
    }
    
    return profile


def batch_profile_users(df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """
    Generate personas for multiple users/records.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Features for all users
    predictions : np.ndarray
        Addiction predictions for all users
    
    Returns:
    --------
    pd.DataFrame : Results with persona assignments
    """
    df_norm = normalize_features(df, BEHAVIORAL_FEATURES + PSYCHOLOGICAL_FEATURES)
    
    results = []
    
    for idx, (_, row) in enumerate(df_norm.iterrows()):
        persona, confidence = classify_user_persona(row)
        results.append({
            "user_id": idx,
            "addiction_score": predictions[idx] if idx < len(predictions) else None,
            "persona": persona,
            "confidence": confidence
        })
    
    return pd.DataFrame(results)
