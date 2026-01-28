import streamlit as st
import pandas as pd
import os

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Smartphone Addiction ML Dashboard",
    layout="wide"
)

st.title("📱 Smartphone Addiction Prediction – ML Dashboard")

# ------------------------------
# Section 1: Project Overview
# ------------------------------
st.header("Project Overview")
st.write("""
This project analyzes **smartphone addiction risk** using machine learning by
comparing **behavioral features**, **psychological features**, and their **combined impact**.

The objective is not diagnosis, but **early risk identification and awareness**,
supporting preventive and educational interventions.
""")

# ------------------------------
# Section 2: Methodology Snapshot
# ------------------------------
st.header("Methodology Summary")

st.markdown("""
**Feature Groups**
- **Behavioral**: Screen time, phone unlocks, night usage, app usage patterns
- **Psychological**: Stress, anxiety, sleep quality, self-control, loneliness

**Machine Learning Models**
- Logistic Regression
- Decision Tree
- Random Forest

**Core Innovation**
- Comparative analysis of feature groups
- Risk-level to action mapping
""")

# ------------------------------
# Section 3: Dataset Preview
# ------------------------------
st.header("Dataset Preview")

DATA_PATH = "data/smartphone_addiction.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    st.success("Dataset loaded successfully.")
    st.write(f"Dataset Shape: {df.shape}")
    st.dataframe(df.head(10))
else:
    st.warning(
        "Dataset not found yet.\n\n"
        "Please place the dataset file at:\n"
        "`data/smartphone_addiction.csv`"
    )

# ------------------------------
# Section 4: Model Comparison Results
# ------------------------------
st.header("Model Performance Comparison")

RESULTS_PATH = "results/feature_comparison_results.csv"

if os.path.exists(RESULTS_PATH):
    results = pd.read_csv(RESULTS_PATH)
    st.success("Experiment results loaded.")
    st.dataframe(results)

    st.subheader("Accuracy Comparison")
    st.bar_chart(
        results.pivot(
            index="Feature Set",
            columns="Model",
            values="Accuracy"
        )
    )
else:
    st.info(
        "Model results not available yet.\n\n"
        "Run the ML experiment (`main.py`) to generate results."
    )

# ------------------------------
# Section 5: Risk-Based Recommendations
# ------------------------------
st.header("Risk-Based Recommendations")

st.markdown("""
| Predicted Risk Level | Suggested Action |
|---------------------|------------------|
| **Low** | Maintain current smartphone usage habits |
| **Moderate** | Reduce night-time usage and enable screen-time reminders |
| **High** | Seek counseling support and restrict notifications and app usage |
""")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption(
    "Note: This dashboard is intended for educational and analytical purposes only. "
    "It does not provide clinical diagnosis."
)
