"""
Time-Series Analyzer for Detecting Addiction Escalation Patterns
Tracks behavioral trends and identifies dangerous trajectory changes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from sklearn.linear_model import LinearRegression


def detect_escalation_patterns(df: pd.DataFrame, behavioral_cols: List[str]) -> Dict:
    """
    Detect escalation patterns in behavioral metrics over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with behavioral columns and temporal order
    behavioral_cols : List[str]
        List of behavioral feature columns to analyze
    
    Returns:
    --------
    Dict : Escalation metrics with trends and risk flags
    """
    escalation_report = {}
    
    for col in behavioral_cols:
        if col in df.columns:
            values = df[col].dropna().values
            
            if len(values) > 2:
                # Linear trend detection
                X = np.arange(len(values)).reshape(-1, 1)
                model = LinearRegression()
                model.fit(X, values)
                slope = model.coef_[0]
                
                # Calculate volatility (standard deviation of changes)
                changes = np.diff(values)
                volatility = np.std(changes) if len(changes) > 0 else 0
                
                # Detect sharp increases
                recent_change = (values[-1] - values[0]) / (values[0] + 1e-6)
                
                escalation_report[col] = {
                    "trend_slope": float(slope),
                    "volatility": float(volatility),
                    "percent_change": float(recent_change * 100),
                    "is_escalating": slope > 0,  # Positive slope = escalation
                    "risk_level": "HIGH" if slope > 0.5 else ("MEDIUM" if slope > 0 else "LOW")
                }
    
    return escalation_report


def calculate_addiction_trajectory(addiction_scores: np.ndarray) -> Dict:
    """
    Analyze the trajectory of addiction scores over time.
    
    Parameters:
    -----------
    addiction_scores : np.ndarray
        Array of addiction prediction scores over time (0-1)
    
    Returns:
    --------
    Dict : Trajectory metrics including velocity and acceleration
    """
    if len(addiction_scores) < 2:
        return {"status": "insufficient_data"}
    
    # Velocity: rate of change in addiction score
    velocity = np.diff(addiction_scores)
    avg_velocity = np.mean(velocity)
    
    # Acceleration: rate of change in velocity
    acceleration = np.diff(velocity) if len(velocity) > 1 else np.array([0])
    avg_acceleration = np.mean(acceleration) if len(acceleration) > 0 else 0
    
    # Trend determination
    if avg_velocity > 0.1:
        trend = "RAPIDLY_ESCALATING"
    elif avg_velocity > 0.02:
        trend = "GRADUALLY_ESCALATING"
    elif avg_velocity < -0.1:
        trend = "IMPROVING"
    else:
        trend = "STABLE"
    
    return {
        "avg_velocity": float(avg_velocity),
        "avg_acceleration": float(avg_acceleration),
        "trend": trend,
        "recent_score": float(addiction_scores[-1]),
        "peak_score": float(np.max(addiction_scores)),
        "baseline_score": float(addiction_scores[0])
    }


def identify_critical_features(df: pd.DataFrame, behavioral_cols: List[str], 
                               psychological_cols: List[str]) -> Dict[str, float]:
    """
    Identify which features are most significantly changing.
    Helps determine the primary driver of addiction risk.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    behavioral_cols : List[str]
        Behavioral feature names
    psychological_cols : List[str]
        Psychological feature names
    
    Returns:
    --------
    Dict : Feature importance based on temporal change
    """
    feature_importance = {}
    
    all_cols = behavioral_cols + psychological_cols
    
    for col in all_cols:
        if col in df.columns:
            values = df[col].dropna().values
            if len(values) > 1:
                # Normalize change to 0-100 scale
                min_val = np.min(values) + 1e-6
                max_val = np.max(values) + 1e-6
                normalized_change = abs(values[-1] - values[0]) / (max_val - min_val)
                feature_importance[col] = float(normalized_change)
    
    # Normalize to sum to 1
    total = sum(feature_importance.values()) + 1e-6
    feature_importance = {k: v / total for k, v in feature_importance.items()}
    
    return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
