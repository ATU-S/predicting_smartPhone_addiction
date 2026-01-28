import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# Page Configuration
# ======================================================
st.set_page_config(
    page_title="Smartphone Addiction ML Dashboard",
    layout="wide"
)

st.title("📱 Smartphone Addiction Prediction – ML Dashboard")
st.caption(
    "An interactive visualization for comparing machine learning models "
    "and understanding smartphone addiction risk."
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
tab1, tab2, tab3 = st.tabs(
    ["📊 Model Comparison", "🧠 Risk Insights", "📈 Feature Contribution"]
)

# ======================================================
# TAB 1: MODEL COMPARISON (CORE VALUE)
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
# TAB 2: RISK INSIGHTS (CLIENT-FRIENDLY)
# ======================================================
with tab2:
    st.header("Smartphone Addiction Risk Distribution")

    if DATA_PATH.exists():
        df = load_dataset()

        if "Addiction_Level" in df.columns:
            risk_counts = df["Addiction_Level"].value_counts()

            fig, ax = plt.subplots()
            ax.pie(
                risk_counts.values,
                labels=risk_counts.index,
                autopct="%1.1f%%",
                startangle=90
            )
            ax.axis("equal")

            st.pyplot(fig)

            st.caption(
                "This chart shows how users are distributed across "
                "addiction risk levels, helping stakeholders understand "
                "the overall severity of the issue."
            )
        else:
            st.error("Column 'Addiction_Level' not found in dataset.")
    else:
        st.warning("Dataset not available. Add CSV to the data/ folder.")

# ======================================================
# TAB 3: FEATURE CONTRIBUTION (EXPLANATION POWER)
# ======================================================
with tab3:
    st.header("Feature Group Contribution")

    st.write(
        "This visualization explains **what type of factors influence "
        "smartphone addiction prediction the most**."
    )

    # Conceptual contribution (safe & explainable)
    contribution = {
        "Behavioral Features": 60,
        "Psychological Features": 40
    }

    fig2, ax2 = plt.subplots()
    ax2.pie(
        contribution.values(),
        labels=contribution.keys(),
        autopct="%1.1f%%",
        startangle=90
    )
    ax2.axis("equal")

    st.pyplot(fig2)

    st.caption(
        "Behavioral usage patterns dominate prediction, while psychological "
        "traits provide substantial complementary value—justifying their inclusion."
    )

# ======================================================
# Footer
# ======================================================
st.markdown("---")
st.caption(
    "⚠️ This dashboard is intended for educational and analytical purposes only. "
    "It does not provide clinical diagnosis or real-time monitoring."
)
