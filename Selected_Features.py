#!/usr/bin/env python3
"""
Script: Selected_Features.py

Description:
    This script loads a binary-feature matrix of SARS-CoV-2 variant data to model
    the global case fatality rate (CFR) using Lasso regression with a pre-chosen
    regularization strength (α = 0.000281). It:
      1. Reads the training feature matrix (dropping SampleID, Variant, Global CFR).
      2. Scales the binary features (zero-mean, unit variance).
      3. Fits a Lasso model to the scaled data.
      4. Identifies and prints the subset of features with non-zero coefficients,
         indicating their relative importance in predicting CFR.

Usage:
    python3 Selected_Features.py

Dependencies:
    - numpy
    - pandas
    - scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# 1) Load the training data
df_train = pd.read_csv("feature_matrix_train.csv")
feature_names = df_train.columns[3:]  # drop SampleID, Variant, Global CFR
X_train = df_train[feature_names].values
y_train = df_train["Global CFR"].values

# 2) Scale the features (binary data)
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)

# 3) Fit Lasso with the chosen alpha
alpha = 0.000174
lasso = Lasso(alpha=alpha, max_iter=10000)
lasso.fit(X_train_scaled, y_train)

# 4) Extract and print the selected features
coefs = lasso.coef_
selected_idx = np.where(coefs != 0)[0]

print(f"Features selected at α={alpha}:")
for idx in selected_idx:
    print(f"  {feature_names[idx]}  →  coefficient = {coefs[idx]:.6f}")
