[README.md](https://github.com/user-attachments/files/31115992/README.md)
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
| Logistic Regression | Performs as a solid linear baseline with 81.77% accuracy. However, recall is very low (0.4461), meaning the model misses more than half of the actual high-income individuals. The dataset has inherently non-linear decision boundaries (e.g., interactions among education, occupation, and capital gain) that a linear model cannot capture. Precision of 0.7143 is reasonable — when it predicts >50K, it is correct ~71% of the time. Fast to train and highly interpretable via feature coefficients. |
| Decision Tree | Achieves 85.08% accuracy — second best overall. With max_depth=10, it captures non-linear feature interactions that Logistic Regression cannot model. Recall improves to 0.6025, correctly identifying 60% of high-income earners. AUC of 0.8843 indicates strong discriminative ability. F1 of 0.6679 reflects a good precision-recall balance. The tree structure is interpretable and easy to explain, though single trees can still overfit compared to ensemble methods. |
| KNN | Achieves 82.41% accuracy — a moderate performer. As an instance-based learner (k=7), it handles non-linearity but struggles with high-cardinality categorical features encoded as ordinal integers, where Euclidean distance becomes less meaningful. Recall of 0.5819 is better than Logistic Regression. A key limitation is slow inference time, as all ~24K training samples must be searched at prediction time, making it unsuitable for large-scale or real-time deployment. |
| Naive Bayes | Weakest model with 79.78% accuracy and F1 of only 0.4485. The Gaussian Naive Bayes assumes feature independence and normal distribution — both strongly violated in this dataset. Census features are highly correlated (e.g., education and education_num, occupation and workclass). The extremely low recall of 0.3302 means it misclassifies ~67% of actual high-income earners. Its only advantage is near-instant training and minimal memory usage. |
| Random Forest (Ensemble) | Best performing model with 85.96% accuracy and the highest scores across all metrics — AUC 0.9142, Precision 0.7768, Recall 0.6119, F1 0.6845, MCC 0.6029. Aggregating 100 decision trees via bagging and random feature subsets reduces variance significantly compared to a single tree. The high AUC indicates superior discrimination across all classification thresholds. Robustly handles mixed feature types, class imbalance, and outliers. Recommended for production deployment. |
| Overall Winner for your dataset? | **Random Forest (Ensemble)** — Achieves the best score across all six metrics (Accuracy: 0.8596, AUC: 0.9142, Precision: 0.7768, Recall: 0.6119, F1: 0.6845, MCC: 0.6029). Its ensemble approach consistently outperforms all individual classifiers, making it the most reliable model for predicting adult income on this dataset. |


