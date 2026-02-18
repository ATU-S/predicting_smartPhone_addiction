# Smartphone Addiction Prediction and Digital Wellbeing Insights

Final year ML project that predicts smartphone addiction risk, compares models, and delivers an interactive dashboard with risk personas, escalation patterns, and an intervention simulator.

## Project Highlights
- Multi-model evaluation across behavioral and psychological feature sets
- Risk personas for targeted interventions
- Escalation pattern detection and critical feature discovery
- Intervention simulator for what-if risk reduction projections
- Streamlit dashboard with performance visuals and personalized insights

## Objectives
1. Predict addiction level from behavioral and psychological indicators.
2. Compare multiple ML models and feature groups.
3. Provide explainability using permutation importance.
4. Translate predictions into actionable digital wellbeing insights.

## Dataset
The dataset is located at `data/smartphone_addiction.csv` with 100 samples and 12 columns.
- Features: screen usage metrics, notifications, sleep quality, stress/anxiety, self-control, loneliness
- Target: `Addiction_Level` with labels `Low`, `Moderate`, `High`

## Methodology
- Preprocessing: fixed label mapping, stratified split
- Models: Logistic Regression, Decision Tree, Random Forest
- Metrics: Accuracy, F1-weighted, F1-macro, Precision-weighted, Recall-weighted, ROC-AUC (OvR)
- Model selection: best by `F1-weighted`
- Explainability: permutation importance on the best model

## Innovation: Intervention Simulator
The dashboard includes a simulator that projects how selected interventions might reduce addiction risk over time. It uses a transparent, diminishing-returns model to visualize expected trends.

## How to Run
1. Install dependencies
```bash
pip install -r requirement.txt
```

2. Train and generate results
```bash
python main.py
```

3. Launch the dashboard
```bash
streamlit run app/app.py
```

## Outputs
- Model comparison results: `results/feature_comparison_results.csv`
- Best model artifacts: `results/artifacts/best_model.joblib`, `results/artifacts/best_model_meta.json`
- Confusion matrix: `results/confusion_matrix.png`
- ROC curve: `results/roc_curve.png`
- Feature importance: `results/feature_importance.csv`
- Metrics summary: `results/metrics_summary.json`

## Folder Structure
- `app/` Streamlit dashboard
- `data/` Dataset
- `results/` Experiment outputs and artifacts
- `src/` Core ML pipeline and analytics
- `main.py` Experiment entry point

## Academic Notes
- This project is intended for educational and analytical purposes.
- It does not provide clinical diagnosis or real-time monitoring.

## Future Work
- Add temporal user data for stronger escalation modeling
- Integrate calibrated probabilities and uncertainty estimates
- Extend interventions with causal models and A/B testing
