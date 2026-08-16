"""
ML Assignment 2 - Model Training Script
Dataset: Adult Census Income (UCI ML Repository)
Task: Binary Classification - Predict income > $50K
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)
import joblib
import json
import os

# ─── 1. Load Adult Income Dataset ────────────────────────────────────────────
print("Downloading Adult Income dataset from UCI...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]
df = pd.read_csv(url, names=columns, sep=r',\s*', na_values='?', engine='python')
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# ─── 2. Preprocessing ────────────────────────────────────────────────────────
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"After dropping NaN: {len(df)} rows")

# Binary target: 1 if income >50K, else 0
df['income'] = (df['income'] == '>50K').astype(int)
print(f"Class distribution:\n{df['income'].value_counts().to_string()}")

X = df.drop('income', axis=1)
y = df['income']

cat_cols = ['workclass', 'education', 'marital_status', 'occupation',
            'relationship', 'race', 'sex', 'native_country']
num_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain',
            'capital_loss', 'hours_per_week']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
    ],
    remainder='drop'
)

# ─── 3. Train/Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# Save test data (for Streamlit CSV upload demo)
test_df = X_test.copy()
test_df['income'] = y_test.values
test_df.to_csv('test_data.csv', index=False)
print(f"Test data saved to test_data.csv ({len(test_df)} rows, {len(test_df.columns)} columns)")

# ─── 4. Define Models ────────────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)

model_configs = {
    'Logistic Regression': {
        'file': 'model/logistic_regression.pkl',
        'clf': LogisticRegression(max_iter=500, random_state=42, solver='lbfgs')
    },
    'Decision Tree': {
        'file': 'model/decision_tree.pkl',
        'clf': DecisionTreeClassifier(random_state=42, max_depth=10)
    },
    'KNN': {
        'file': 'model/knn.pkl',
        'clf': KNeighborsClassifier(n_neighbors=7, algorithm='ball_tree')
    },
    'Naive Bayes': {
        'file': 'model/naive_bayes.pkl',
        'clf': GaussianNB()
    },
    'Random Forest': {
        'file': 'model/random_forest.pkl',
        'clf': RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, max_depth=15
        )
    }
}

# ─── 5. Train, Evaluate, and Save ────────────────────────────────────────────
metrics_dict = {}

for name, config in model_configs.items():
    print(f"\nTraining {name}...")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', config['clf'])
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
        'AUC':      round(float(roc_auc_score(y_test, y_proba)), 4),
        'Precision':round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'Recall':   round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'F1':       round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'MCC':      round(float(matthews_corrcoef(y_test, y_pred)), 4)
    }
    metrics_dict[name] = metrics
    print(f"  {metrics}")

    joblib.dump(pipeline, config['file'], compress=3)
    size_mb = os.path.getsize(config['file']) / 1e6
    print(f"  Saved → {config['file']} ({size_mb:.1f} MB)")

# Save metrics summary
with open('model/metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=2)
print("\n✓ Metrics saved to model/metrics.json")

# ─── 6. Print Summary Table ──────────────────────────────────────────────────
print("\n" + "=" * 80)
print("METRICS SUMMARY")
print("=" * 80)
summary = pd.DataFrame(metrics_dict).T
summary.index.name = 'Model'
print(summary.to_string())
print("=" * 80)

best_model = summary['Accuracy'].idxmax()
print(f"\nBest model by Accuracy: {best_model}")
print("\nTraining complete!")
