# Adult Census Income Classification — ML Assignment 2

## Problem Statement

The goal is to predict whether an individual's annual income **exceeds $50,000** based on census demographic and employment data. This is a **binary classification** problem where the target variable `income` takes the value `1` (>50K) or `0` (≤50K). Five classification models — Logistic Regression, Decision Tree, KNN, Naive Bayes, and Random Forest — are trained and compared across six evaluation metrics.

---

## Dataset Description

**Name**: Adult Census Income | **Source**: UCI Machine Learning Repository  
**URL**: https://archive.ics.uci.edu/ml/datasets/adult

| Property | Value |
|---|---|
| Instances | 30,162 (after removing 2,399 rows with missing values) |
| Features | 14 (6 numerical, 8 categorical) |
| Target | `income` — 0 (≤50K, 75.1%) / 1 (>50K, 24.9%) |
| Train / Test Split | 24,129 / 6,033 (80/20, stratified) |

**Features**: age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country.

**Preprocessing**: Missing values dropped · Numerical features scaled with `StandardScaler` · Categorical features encoded with `OrdinalEncoder` · All steps wrapped in an `sklearn.Pipeline`.

---

## GitHub Repository Link
[https://github.com/ashishagr25/ml_assignment_2](https://github.com/ashishagr25/ml_assignment_2)

---

## Streamlit App Link
[https://mlassignment2-asgdmxjp4wbtt9immin6kw.streamlit.app/](https://mlassignment2-asgdmxjp4wbtt9immin6kw.streamlit.app/)

---

## Models Used

### Comparison Table (Evaluated on 20% hold-out test set — 6,033 samples)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8177 | 0.8500 | 0.7143 | 0.4461 | 0.5492 | 0.4617 |
| Decision Tree | 0.8508 | 0.8843 | 0.7492 | 0.6025 | 0.6679 | 0.5788 |
| KNN | 0.8241 | 0.8574 | 0.6687 | 0.5819 | 0.6223 | 0.5105 |
| Naive Bayes | 0.7978 | 0.8498 | 0.6986 | 0.3302 | 0.4485 | 0.3798 |
| Random Forest (Ensemble) | **0.8596** | **0.9142** | **0.7768** | **0.6119** | **0.6845** | **0.6029** |

> Bold values indicate the best score per metric.

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Good baseline with 81.77% accuracy but low recall (0.4461) — misses many high-income individuals due to non-linear decision boundaries in the data that a linear model cannot capture. |
| Decision Tree | Second best at 85.08% accuracy. Captures non-linear feature interactions (max_depth=10) with better recall (0.6025) and strong AUC (0.8843). Interpretable but prone to overfitting compared to ensemble methods. |
| kNN | Moderate performer at 82.41% accuracy. Distance-based learning is weakened by ordinal-encoded categorical features. Recall (0.5819) is reasonable but inference is slow over ~24K training samples. |
| Naive Bayes | Weakest model at 79.78% accuracy and F1 of 0.4485. Gaussian independence assumption is violated by correlated census features, resulting in very low recall (0.3302). Fast to train but unsuitable for this dataset. |
| Random Forest (Ensemble) | Best overall — 85.96% accuracy, highest AUC (0.9142), F1 (0.6845), and MCC (0.6029). Bagging over 100 trees reduces variance and handles mixed feature types and class imbalance robustly. |
| Overall Winner for your dataset? | **Random Forest (Ensemble)** — top performer across all six metrics, making it the most reliable model for this dataset. |
| Overall Winner for your dataset? | **Random Forest (Ensemble)** — Achieves the best score across all six metrics (Accuracy: 0.8596, AUC: 0.9142, Precision: 0.7768, Recall: 0.6119, F1: 0.6845, MCC: 0.6029). Its ensemble approach consistently outperforms all individual classifiers, making it the most reliable model for predicting adult income on this dataset. |


