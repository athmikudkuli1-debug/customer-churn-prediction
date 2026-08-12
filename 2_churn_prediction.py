"""
CUSTOMER CHURN PREDICTION — full beginner-friendly pipeline

What this script does, step by step:
  1. Load the data
  2. Explore it (EDA) — understand what we're working with
  3. Preprocess it — turn raw data into numbers a model can learn from
  4. Train two classification models (Logistic Regression, Random Forest)
  5. Evaluate them (accuracy, precision, recall, F1, ROC-AUC)
  6. Look at WHICH features matter most for churn

Every section has comments explaining WHY, not just WHAT.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # so plots save to file instead of trying to pop up a window
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

sns.set_style("whitegrid")
OUT = "outputs"

# ============================================================
# STEP 1: LOAD THE DATA
# ============================================================
df = pd.read_csv("customer_churn.csv")
print("Shape:", df.shape)
print(df.head())
print("\nMissing values per column:\n", df.isnull().sum())

# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
# Why EDA matters: before training any model, you need to understand
# your data — what the churn rate looks like, which features seem
# related to churn, whether there are data quality issues, etc.
# Skipping this step is the #1 mistake beginners make.

# 2a. Overall churn rate
churn_rate = (df["Churn"] == "Yes").mean()
print(f"\nOverall churn rate: {churn_rate:.1%}")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Churn count
sns.countplot(data=df, x="Churn", ax=axes[0, 0], palette=["#4C72B0", "#DD8452"])
axes[0, 0].set_title("Customers: Churned vs Stayed")

# Churn rate by contract type — a classic, very predictive feature
churn_by_contract = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean())
churn_by_contract.plot(kind="bar", ax=axes[0, 1], color="#55A868")
axes[0, 1].set_title("Churn Rate by Contract Type")
axes[0, 1].set_ylabel("Churn Rate")
axes[0, 1].tick_params(axis="x", rotation=20)

# Tenure distribution split by churn
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Tenure Distribution by Churn")

# Monthly charges by churn
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1, 1])
axes[1, 1].set_title("Monthly Charges by Churn")

plt.tight_layout()
plt.savefig(f"{OUT}/1_eda_overview.png", dpi=120)
plt.close()
print(f"Saved EDA overview chart to {OUT}/1_eda_overview.png")

# ============================================================
# STEP 3: PREPROCESSING
# ============================================================
# Models need numbers, not text. We convert categorical columns
# ("Yes"/"No", "Month-to-month", etc.) into numeric form.

model_df = df.drop(columns=["customerID"])  # not predictive, just an ID

# Target variable: Churn -> 1/0
model_df["Churn"] = model_df["Churn"].map({"Yes": 1, "No": 0})

# Identify categorical columns to encode
categorical_cols = model_df.select_dtypes(include="object").columns.tolist()
print("\nCategorical columns to encode:", categorical_cols)

# One-hot encoding: turns e.g. Contract into Contract_Month-to-month,
# Contract_One year, Contract_Two year columns of 0/1.
# We use drop_first=True to avoid redundant columns (avoids multicollinearity).
model_df = pd.get_dummies(model_df, columns=categorical_cols, drop_first=True)

X = model_df.drop(columns=["Churn"])
y = model_df["Churn"]

# Train/test split: we hold out 25% of customers the model never sees
# during training, so we can honestly evaluate how well it generalizes.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Scaling: Logistic Regression is sensitive to feature scale
# (e.g. tenure ranges 0-72, MonthlyCharges ranges 18-120).
# Random Forest doesn't need this, but scaling doesn't hurt it either.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# STEP 4: TRAIN MODELS
# ============================================================

# --- Model A: Logistic Regression ---
# A simple, interpretable baseline. Good starting point for any
# classification problem before trying more complex models.
#
# IMPORTANT — class imbalance: only ~29% of customers churn. A model
# that just predicted "No churn" for everyone would already be ~71%
# accurate while catching ZERO real churners — useless for the business,
# since the whole point is to find people about to leave. We use
# class_weight="balanced" so the model pays more attention to the
# minority (churn) class instead of optimizing for raw accuracy.
log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

# --- Model B: Random Forest ---
# An ensemble of decision trees. Usually more accurate than logistic
# regression on tabular data, and gives us feature importances.
rf = RandomForestClassifier(
    n_estimators=300, max_depth=6, random_state=42, class_weight="balanced"
)
rf.fit(X_train, y_train)  # tree models don't need scaled features
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# ============================================================
# STEP 5: EVALUATE MODELS
# ============================================================
def evaluate(name, y_true, y_pred, y_proba):
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
    print(f"F1 score:  {f1_score(y_true, y_pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.3f}")
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }

results = []
results.append(evaluate("Logistic Regression", y_test, y_pred_lr, y_proba_lr))
results.append(evaluate("Random Forest", y_test, y_pred_rf, y_proba_rf))

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT}/model_comparison.csv", index=False)

# Confusion matrices — show exactly what the model got right/wrong
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, y_pred) in zip(axes, [("Logistic Regression", y_pred_lr), ("Random Forest", y_pred_rf)]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    ax.set_title(f"Confusion Matrix: {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT}/2_confusion_matrices.png", dpi=120)
plt.close()

# ROC curves
plt.figure(figsize=(6, 5))
for name, y_proba in [("Logistic Regression", y_proba_lr), ("Random Forest", y_proba_rf)]:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/3_roc_curves.png", dpi=120)
plt.close()

# ============================================================
# STEP 6: FEATURE IMPORTANCE
# ============================================================
# Which factors actually drive churn? This is often more valuable
# to a business than the prediction itself — it tells them WHAT to fix.
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 features driving churn (Random Forest):")
print(importances.head(10))

plt.figure(figsize=(8, 6))
importances.head(10).sort_values().plot(kind="barh", color="#C44E52")
plt.title("Top 10 Features Driving Churn (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/4_feature_importance.png", dpi=120)
plt.close()

print(f"\nAll charts saved to {OUT}/")
print("\nDone.")
