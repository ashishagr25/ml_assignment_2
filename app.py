"""
ML Assignment 2 - Streamlit Web Application
Adult Census Income Classification
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adult Income Classification",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom Styling ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("💰 Adult Income Classification")
st.markdown("**Dataset**: Adult Census Income (UCI ML Repository) | "
            "**Task**: Predict whether annual income exceeds $50K")
st.divider()

# ─── Model File Mapping ───────────────────────────────────────────────────────
MODEL_FILES = {
    'Logistic Regression':    'model/logistic_regression.pkl',
    'Decision Tree':          'model/decision_tree.pkl',
    'K-Nearest Neighbor (KNN)': 'model/knn.pkl',
    'Naive Bayes (Gaussian)': 'model/naive_bayes.pkl',
    'Random Forest (Ensemble)': 'model/random_forest.pkl',
}

EXPECTED_COLS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'
]

# ─── Cached Loaders ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_precomputed_metrics():
    metrics_path = 'model/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return {}

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

# CSV upload
# Default dataset path
DEFAULT_CSV = "test_data.csv"

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Test Data (CSV)",
    type=["csv"],
    help="Upload your own CSV or use the default test_data.csv"
)
# Use uploaded file if provided, otherwise use default CSV
if uploaded_file is not None:
    df_source = uploaded_file
    st.sidebar.success("Using uploaded file")
else:
    df_source = DEFAULT_CSV
    st.sidebar.info("Using default file: test_data.csv")

# Model selection dropdown
selected_model_name = st.sidebar.selectbox(
    "🤖 Select Classification Model",
    list(MODEL_FILES.keys()),
    index=0
)

st.sidebar.divider()
st.sidebar.markdown("**Expected CSV columns (14 features + target):**")
st.sidebar.code(
    "age, workclass, fnlwgt, education,\n"
    "education_num, marital_status, occupation,\n"
    "relationship, race, sex, capital_gain,\n"
    "capital_loss, hours_per_week,\n"
    "native_country, income"
)
st.sidebar.markdown("💡 Use `test_data.csv` from the GitHub repo as sample data.")

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Model Evaluation", "📈 All Models Comparison", "ℹ️ About Dataset"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Model Evaluation on Uploaded Data
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    try:
        df = pd.read_csv(df_source)
        st.success(f"✅ Dataset loaded: **{len(df)} rows × {len(df.columns)} columns**")

        with st.expander("🔍 Preview Uploaded Data (first 10 rows)"):
            st.dataframe(df.head(10), use_container_width=True)

        if 'income' not in df.columns:
            st.error("❌ The uploaded CSV must contain an **'income'** column (0 = ≤50K, 1 = >50K).")
        else:
            X = df.drop('income', axis=1)
            y = df['income'].astype(int)

            # Validate feature columns
            missing_cols = [c for c in EXPECTED_COLS if c not in X.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.stop()

            model_path = MODEL_FILES[selected_model_name]
            if not os.path.exists(model_path):
                st.error(f"Model file not found: `{model_path}`. "
                         "Run `train_models.py` first to train and save the models.")
                st.stop()

            with st.spinner(f"Running {selected_model_name}..."):
                model = load_model(model_path)
                y_pred  = model.predict(X)
                y_proba = model.predict_proba(X)[:, 1]

            # ── Evaluation Metrics ──
            metrics = {
                'Accuracy':  round(float(accuracy_score(y, y_pred)), 4),
                'AUC Score': round(float(roc_auc_score(y, y_proba)), 4),
                'Precision': round(float(precision_score(y, y_pred, zero_division=0)), 4),
                'Recall':    round(float(recall_score(y, y_pred, zero_division=0)), 4),
                'F1 Score':  round(float(f1_score(y, y_pred, zero_division=0)), 4),
                'MCC':       round(float(matthews_corrcoef(y, y_pred)), 4)
            }

            st.subheader(f"📊 Evaluation Metrics — {selected_model_name}")
            metric_cols = st.columns(6)
            for i, (mname, val) in enumerate(metrics.items()):
                metric_cols[i].metric(label=mname, value=f"{val:.4f}")

            st.divider()
            col_cm, col_cr = st.columns(2)

            # ── Confusion Matrix ──
            with col_cm:
                st.subheader("🔢 Confusion Matrix")
                cm = confusion_matrix(y, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['≤50K (0)', '>50K (1)'],
                    yticklabels=['≤50K (0)', '>50K (1)']
                )
                ax_cm.set_xlabel('Predicted Label', fontsize=11)
                ax_cm.set_ylabel('True Label', fontsize=11)
                ax_cm.set_title(f'{selected_model_name}', fontsize=12)
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close(fig_cm)

            # ── Classification Report ──
            with col_cr:
                st.subheader("📋 Classification Report")
                report_dict = classification_report(
                    y, y_pred,
                    target_names=['≤50K (0)', '>50K (1)'],
                    output_dict=True
                )
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.round(4), use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")

    
    st.markdown("""
    ### How to Use
    1. **Upload** `test_data.csv` (from GitHub repo) via the sidebar
    2. **Select** a model from the dropdown
    3. View **evaluation metrics**, **confusion matrix**, and **classification report**

    ### Dataset Features
    | Feature | Type | Description |
    |---------|------|-------------|
    | age | Numerical | Age of the individual |
    | workclass | Categorical | Employment type |
    | fnlwgt | Numerical | Census weight |
    | education | Categorical | Education level |
    | education_num | Numerical | Education years |
    | marital_status | Categorical | Marital status |
    | occupation | Categorical | Job type |
    | relationship | Categorical | Relationship status |
    | race | Categorical | Race |
    | sex | Categorical | Gender |
    | capital_gain | Numerical | Capital gains |
    | capital_loss | Numerical | Capital losses |
    | hours_per_week | Numerical | Work hours/week |
    | native_country | Categorical | Country of origin |
    | **income** | **Target** | **0=≤50K, 1=>50K** |
    """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: All Models Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📈 All Models Performance Comparison")
    all_metrics = load_precomputed_metrics()

    if all_metrics:
        # ── Metrics Table ──
        metrics_df = pd.DataFrame(all_metrics).T.reset_index()
        metrics_df.rename(columns={'index': 'Model'}, inplace=True)
        metrics_df.set_index('Model', inplace=True)

        st.markdown("**Metrics on hold-out test set (20% of dataset)**")
        styled_df = metrics_df.style.highlight_max(
            subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
            color='#c6efce'
        ).highlight_min(
            subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
            color='#ffc7ce'
        ).format("{:.4f}")
        st.dataframe(styled_df, use_container_width=True)

        # ── Best Model Highlight ──
        best_acc = metrics_df['Accuracy'].idxmax()
        best_f1  = metrics_df['F1'].idxmax()
        best_auc = metrics_df['AUC'].idxmax()
        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.success(f"🏆 Best Accuracy: **{best_acc}** ({metrics_df.loc[best_acc, 'Accuracy']:.4f})")
        col_b2.success(f"🏆 Best F1 Score: **{best_f1}** ({metrics_df.loc[best_f1, 'F1']:.4f})")
        col_b3.success(f"🏆 Best AUC: **{best_auc}** ({metrics_df.loc[best_auc, 'AUC']:.4f})")

        st.divider()

        # ── Bar Chart Comparison ──
        st.subheader("Bar Chart Comparison Across All Metrics")
        metric_cols = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        palette = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
        model_labels = [
            m.replace('K-Nearest Neighbor (KNN)', 'KNN')
             .replace('Random Forest (Ensemble)', 'Random\nForest')
             .replace('Naive Bayes (Gaussian)', 'Naive\nBayes')
             .replace('Decision Tree', 'Decision\nTree')
             .replace('Logistic Regression', 'Logistic\nReg')
            for m in metrics_df.index
        ]

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for idx, metric in enumerate(metric_cols):
            ax = axes[idx // 3][idx % 3]
            values = metrics_df[metric].values
            bars = ax.bar(model_labels, values, color=palette,
                          edgecolor='black', linewidth=0.4, width=0.6)
            ax.set_title(metric, fontsize=13, fontweight='bold', pad=6)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel('Score', fontsize=9)
            ax.tick_params(axis='x', labelsize=8)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        fig.suptitle('Model Comparison — Adult Income Classification',
                     fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Observations Table ──
        st.divider()
        st.subheader("Model Observations")
        observations = {
            'Model': [
                'Logistic Regression', 'Decision Tree', 'KNN',
                'Naive Bayes (Gaussian)', 'Random Forest (Ensemble)', 'Overall Winner'
            ],
            'Observation': [
                f"Accuracy: {all_metrics.get('Logistic Regression', {}).get('Accuracy', 'N/A')} — "
                "Solid baseline with fast training. Good precision; slightly lower recall on minority class (>50K). "
                "Performs well on linearly separable patterns in this dataset.",
                f"Accuracy: {all_metrics.get('Decision Tree', {}).get('Accuracy', 'N/A')} — "
                "Moderate performance; prone to overfitting despite max_depth=10. "
                "Interpretable structure but less robust than ensemble methods.",
                f"Accuracy: {all_metrics.get('KNN', {}).get('Accuracy', 'N/A')} — "
                "Competitive accuracy but sensitive to scale and high-cardinality categorical features. "
                "Slower inference due to instance-based learning on ~38K samples.",
                f"Accuracy: {all_metrics.get('Naive Bayes', {}).get('Accuracy', 'N/A')} — "
                "Weakest performer due to Gaussian independence assumption violated by correlated features. "
                "Very fast training and low memory footprint.",
                f"Accuracy: {all_metrics.get('Random Forest', {}).get('Accuracy', 'N/A')} — "
                "Best overall performance. Handles mixed feature types well, robust to outliers, "
                "and reduces overfitting via bagging. Highest AUC and MCC scores.",
                "Random Forest (Ensemble) — Consistently top performer across all six metrics "
                "(Accuracy, AUC, Precision, Recall, F1, MCC). Recommended for production deployment."
            ]
        }
        obs_df = pd.DataFrame(observations).set_index('Model')
        st.dataframe(obs_df, use_container_width=True)

    else:
        st.warning("⚠️ No pre-computed metrics found. Run `train_models.py` to train all models first.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: About Dataset
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("ℹ️ About the Adult Census Income Dataset")
    st.markdown("""
    **Source**: UCI Machine Learning Repository  
    **URL**: https://archive.ics.uci.edu/ml/datasets/adult  
    **Donor**: Ronny Kohavi and Barry Becker, 1996

    ### Problem Statement
    Predict whether an individual earns more than \\$50,000 per year based on census demographic attributes.
    This is a **binary classification** problem: **income > 50K (1)** vs **income ≤ 50K (0)**.

    ### Dataset Statistics
    | Property | Value |
    |----------|-------|
    | Total Instances | ~48,842 |
    | Features | 14 |
    | Numerical Features | 6 |
    | Categorical Features | 8 |
    | Target Classes | 2 (Binary) |
    | Missing Values | ~7% (rows dropped) |
    | Class Balance | ~76% ≤50K / ~24% >50K |

    ### Feature Description
    | Feature | Type | Description |
    |---------|------|-------------|
    | age | Continuous | Age in years |
    | workclass | Categorical | Employment sector |
    | fnlwgt | Continuous | Census sampling weight |
    | education | Categorical | Highest education level |
    | education_num | Continuous | Numeric education years |
    | marital_status | Categorical | Marital status |
    | occupation | Categorical | Type of occupation |
    | relationship | Categorical | Relationship role |
    | race | Categorical | Race |
    | sex | Categorical | Biological sex |
    | capital_gain | Continuous | Capital gains from investments |
    | capital_loss | Continuous | Capital losses from investments |
    | hours_per_week | Continuous | Hours worked per week |
    | native_country | Categorical | Country of birth |

    ### Preprocessing Applied
    - **Missing values**: Rows with '?' entries dropped (~3.8K rows removed)
    - **Numerical features**: StandardScaler (zero mean, unit variance)
    - **Categorical features**: OrdinalEncoder (integer encoding)
    - **Train/Test split**: 80% / 20%, stratified by target class
    """)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<small>ML Assignment 2 | M.Tech AIML/DSE | BITS Pilani WILP | "
    "Adult Census Income Dataset (UCI ML Repository)</small>",
    unsafe_allow_html=True
)
