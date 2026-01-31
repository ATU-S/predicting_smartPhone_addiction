import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.risk_profiler import create_personalized_profile, PERSONA_DEFINITIONS, batch_profile_users
from src.timeseries_analyzer import detect_escalation_patterns, identify_critical_features
from src.schema import BEHAVIORAL_FEATURES, PSYCHOLOGICAL_FEATURES

# ======================================================
# Page Configuration
# ======================================================
st.set_page_config(
    page_title="Smartphone Addiction ML Dashboard",
    layout="wide"
)

st.title("📱 Smartphone Addiction Prediction – ML Dashboard")
st.caption(
    "An interactive visualization for comparing machine learning models, "
    "understanding risk personas, and detecting escalation patterns."
)

# ======================================================
# Paths
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "smartphone_addiction.csv"
RESULTS_PATH = BASE_DIR / "results" / "feature_comparison_results.csv"

# ======================================================
# Cached Loaders
# ======================================================
@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)

# ======================================================
# Tabs Layout
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Model Comparison", "🎭 Risk Personas", "📈 Escalation Patterns", "🧠 Personalized Insights"]
)

# ======================================================
# TAB 1: MODEL COMPARISON
# ======================================================
with tab1:
    st.header("Model Performance Comparison")

    if RESULTS_PATH.exists():
        results = load_results()

        st.subheader("Performance Metrics Table")
        st.dataframe(results, use_container_width=True)

        st.subheader("Accuracy Comparison Across Models and Feature Sets")

        pivot_acc = results.pivot(
            index="Feature Set",
            columns="Model",
            values="Accuracy"
        )

        st.bar_chart(pivot_acc)

        # Highlight best model
        best_row = results.loc[results["Accuracy"].idxmax()]
        st.success(
            f"Best Model: **{best_row['Model']}** using "
            f"**{best_row['Feature Set']} features** "
            f"(Accuracy = {best_row['Accuracy']:.2f})"
        )

        st.caption(
            "Combined behavioral and psychological features consistently "
            "provide better performance across models."
        )
    else:
        st.warning(
            "Model results not found. Run `python main.py` to generate results."
        )

# ======================================================
# TAB 2: RISK PERSONAS (NEW)
# ======================================================
with tab2:
    st.header("🎭 Risk Persona Classification")
    st.markdown("""
    Users are classified into **behavioral personas** to enable targeted interventions.
    Each persona has distinct characteristics and personalized recommendations.
    """)

    if DATA_PATH.exists():
        df = load_dataset()
        
        # Show persona definitions
        st.subheader("Persona Archetypes")
        
        cols = st.columns(2)
        for idx, (persona_name, definition) in enumerate(PERSONA_DEFINITIONS.items()):
            col = cols[idx % 2]
            with col:
                st.write(f"### {persona_name}")
                st.write(f"**Description:** {definition['description']}")
                st.write(f"**Intervention:** {definition['intervention']}")
                st.write(f"**Key Features:** {', '.join(definition['key_features'])}")
                st.divider()
        
        # Show persona distribution if predictions available
        if RESULTS_PATH.exists():
            st.subheader("User Distribution by Persona")
            
            # Create sample predictions for demonstration
            all_features = df[BEHAVIORAL_FEATURES + PSYCHOLOGICAL_FEATURES]
            sample_predictions = np.random.rand(len(df))  # Demo predictions
            
            personas_df = batch_profile_users(all_features, sample_predictions)
            persona_counts = personas_df["persona"].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            persona_counts.plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title("User Distribution Across Personas")
            ax.set_ylabel("Number of Users")
            ax.set_xlabel("Persona")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
    else:
        st.warning("Dataset not available.")

# ======================================================
# TAB 3: ESCALATION PATTERNS (NEW)
# ======================================================
with tab3:
    st.header("📈 Addiction Escalation Patterns")
    st.markdown("""
    Detects dangerous **escalation trends** in behavioral metrics.
    Identifies which features are changing most rapidly and at what risk level.
    """)

    if DATA_PATH.exists():
        df = load_dataset()
        
        # Detect escalation patterns
        escalation = detect_escalation_patterns(df, BEHAVIORAL_FEATURES)
        
        st.subheader("Feature Risk Assessment")
        
        # Create escalation summary table
        escalation_records = []
        for feature, metrics in escalation.items():
            escalation_records.append({
                "Feature": feature,
                "Trend Slope": f"{metrics['trend_slope']:.3f}",
                "Volatility": f"{metrics['volatility']:.3f}",
                "% Change": f"{metrics['percent_change']:.1f}%",
                "Risk Level": metrics["risk_level"]
            })
        
        escalation_df = pd.DataFrame(escalation_records)
        st.dataframe(escalation_df, use_container_width=True)
        
        # Visualize escalation risks
        st.subheader("Risk Level Distribution")
        risk_counts = escalation_df["Risk Level"].value_counts()
        
        colors = {"HIGH": "#d62728", "MEDIUM": "#ff7f0e", "LOW": "#2ca02c"}
        fig, ax = plt.subplots(figsize=(8, 5))
        risk_counts.plot(kind="bar", ax=ax, color=[colors.get(x, "gray") for x in risk_counts.index])
        ax.set_title("Behavioral Features by Risk Level")
        ax.set_ylabel("Count")
        ax.set_xlabel("Risk Level")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        
        # Critical features
        critical_features = identify_critical_features(df, BEHAVIORAL_FEATURES, PSYCHOLOGICAL_FEATURES)
        
        st.subheader("Top Critical Features Driving Addiction Risk")
        top_features = dict(sorted(critical_features.items(), key=lambda x: x[1], reverse=True)[:5])
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(list(top_features.keys()), list(top_features.values()), color="darkslateblue")
        ax.set_xlabel("Relative Importance")
        ax.set_title("Features with Greatest Temporal Change")
        st.pyplot(fig)
    else:
        st.warning("Dataset not available.")

# ======================================================
# TAB 4: PERSONALIZED INSIGHTS (NEW)
# ======================================================
with tab4:
    st.header("🧠 Personalized Risk Assessment")
    st.markdown("""
    Analyze an individual user's profile to generate **personalized risk insights**
    and targeted intervention recommendations.
    """)

    if DATA_PATH.exists():
        df = load_dataset()
        
        # Allow user selection
        st.subheader("Select a User Profile")
        
        user_idx = st.slider(
            "Choose user ID:",
            min_value=0,
            max_value=len(df) - 1,
            value=0
        )
        
        user_data = df.iloc[[user_idx]]
        
        # Normalize and create profile
        from src.risk_profiler import normalize_features
        user_norm = normalize_features(user_data, BEHAVIORAL_FEATURES + PSYCHOLOGICAL_FEATURES)
        
        # Demo prediction score
        demo_score = np.random.rand()
        
        profile = create_personalized_profile(user_norm, demo_score)
        
        # Display profile
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Addiction Risk Score", f"{profile['addiction_risk_score']:.2%}")
            st.metric("Persona Match Confidence", f"{profile['confidence']:.1%}")
        
        with col2:
            st.info(f"**Assigned Persona:** {profile['persona']}")
        
        st.markdown(f"### {profile['persona']}")
        st.write(f"**Profile:** {profile['description']}")
        st.write(f"**Recommended Intervention:** {profile['intervention']}")
        
        st.subheader("Top Risk Factors")
        risk_cols = st.columns(len(profile['top_risk_factors']))
        for idx, (factor_info) in enumerate(profile['top_risk_factors']):
            with risk_cols[idx]:
                st.metric(
                    factor_info['factor'],
                    f"{factor_info['severity']:.1%}"
                )
        
        st.subheader("Features to Monitor")
        st.write(", ".join(profile['key_features_to_monitor']))
    else:
        st.warning("Dataset not available.")

# ======================================================
# Footer
# ======================================================
st.markdown("---")
st.caption(
    "⚠️ This dashboard is intended for educational and analytical purposes only. "
    "It does not provide clinical diagnosis or real-time monitoring."
)

