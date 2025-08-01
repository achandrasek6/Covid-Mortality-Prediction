#!/usr/bin/env python3
"""
Updated ML_model.py

Adds persistence so the trained model and scaler can be reused for prediction-only.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

# 1) Load the data
df_train = pd.read_csv("feature_matrix_train.csv")
df_test  = pd.read_csv("feature_matrix_test.csv")

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
os.makedirs("model_artifacts", exist_ok=True)
joblib.dump(lasso_final, os.path.join("model_artifacts", "lasso_model.joblib"))
joblib.dump(scaler, os.path.join("model_artifacts", "scaler.joblib"))

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
