#!/usr/bin/env python3
"""
ML_model.py

Train the Lasso regression model on the precomputed feature matrices and persist the
model + scaler for later prediction/inference. Also reports performance and selected features.

Main responsibilities:
  1. Load the training and testing feature matrices (CSV), which include sample identifiers,
     variant features, and the target variable ("Global CFR").
  2. Scale the features with a StandardScaler (configured to match original training: with_mean=False).
  3. Fit a Lasso model with a specified regularization strength (alpha).
  4. Persist the trained model and scaler to disk under model_artifacts/ for reuse in inference.
  5. Evaluate the model on both training and testing sets, reporting R² and MSE.
  6. Extract and print the nonzero (selected) features and their coefficients.

Expected inputs (by default or via arguments):
  - Training matrix: feature_matrix_train.csv
  - Testing matrix: feature_matrix_test.csv

Outputs:
  - model_artifacts/lasso_model.joblib   : Serialized Lasso model.
  - model_artifacts/scaler.joblib        : Serialized scaler used to transform input features.
  - Console/log output with training/testing performance and retained feature coefficients.

Typical usage (if refactored to CLI):
  python3 scripts/ML_model.py \
    --train-matrix training/feature_matrix_train.csv \
    --test-matrix training/feature_matrix_test.csv \
    --out-dir model_artifacts

Note:
  If you already have the serialized model and scaler saved, this script does not need to be
  re-run for prediction; downstream inference should use those artifacts directly (e.g.,
  via collapse_and_predict.py).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

# 1) Load the data
df_train = pd.read_csv("../lasso_training_data/feature_matrix_train.csv")
df_test  = pd.read_csv("../lasso_training_data/feature_matrix_test.csv")

feature_cols = df_train.columns.drop(["SampleID", "Variant", "Global CFR"])
X_train = df_train[feature_cols].values
y_train = df_train["Global CFR"].values

X_test  = df_test[feature_cols].values
y_test  = df_test["Global CFR"].values

# 2) Scale features
scaler = StandardScaler(with_mean=False)
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 3) Fit Lasso with original α = 0.000174
alpha = 0.000174
lasso_final = Lasso(alpha=alpha, max_iter=10000)
lasso_final.fit(X_train_s, y_train)

# Persist model and scaler for reuse
os.makedirs("../model_artifacts", exist_ok=True)
joblib.dump(lasso_final, os.path.join("../model_artifacts", "lasso_model.joblib"))
joblib.dump(scaler, os.path.join("../model_artifacts", "scaler.joblib"))

# 4) Evaluate on training and testing sets
y_train_pred = lasso_final.predict(X_train_s)
y_test_pred  = lasso_final.predict(X_test_s)

print(f"Training R²: {r2_score(y_train, y_train_pred):.3f}")
print(f"Testing  R²: {r2_score(y_test,  y_test_pred):.3f}")
print(f"Test MSE    : {mean_squared_error(y_test, y_test_pred):.3e}")

# 5) Extract the selected features
coefs = lasso_final.coef_
selected_idx = np.where(coefs != 0)[0]
selected_features = feature_cols[selected_idx]
selected_coefs    = coefs[selected_idx]

print(f"\nNumber of features retained: {len(selected_features)}")
print("Selected features and coefficients:")
for feat, coef in zip(selected_features, selected_coefs):
    print(f"  {feat:15s} → {coef:.6f}")
