import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config("Smartphone Addiction ML", layout="wide")

st.title("📱 Smartphone Addiction – ML Analysis Dashboard")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "smartphone_addiction.csv"
RESULTS_PATH = BASE_DIR / "results" / "feature_comparison_results.csv"

st.header("Project Overview")
st.write("""
This system evaluates smartphone addiction risk using machine learning.
It compares behavioral features, psychological features, and their combined impact.
""")

st.header("Dataset")
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    st.dataframe(df.head())
else:
    st.warning("Dataset not available")

st.header("Model Results")
if RESULTS_PATH.exists():
    results = pd.read_csv(RESULTS_PATH)
    st.dataframe(results)
    st.bar_chart(results.pivot(index="Feature Set", columns="Model", values="Accuracy"))
else:
    st.info("Run main.py to generate results")

st.header("Risk Interpretation")
st.markdown("""
- **Low Risk** → Maintain healthy usage habits  
- **Moderate Risk** → Reduce night-time usage  
- **High Risk** → Seek guidance and limit notifications  
""")
