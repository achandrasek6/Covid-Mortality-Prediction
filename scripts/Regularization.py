#!/usr/bin/env python3
"""
Script: Regularization.py

Description:
    This script evaluates the impact of Lasso regularization strength on feature selection
    and predictive performance when modeling the global case fatality rate (CFR). It:
      1. Loads training and test feature matrices (dropping non-predictive columns).
      2. Scales the features using a zero-mean transformer.
      3. Performs a grid search over a log‐spaced range of Lasso alpha values (1e-5 to 1e-2).
      4. Records, for each alpha:
           - The number of non-zero coefficients (selected features).
           - The test-set R² score.
      5. Outputs a DataFrame summarizing results and generates an “elbow plot” of
         n_features vs. test R² to guide alpha selection.

Usage:
    python3 Regularization.py

Dependencies:
    - numpy
    - pandas
    - scikit-learn
    - matplotlib
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# Load pre‑scaled data (or scale here as before)
X_train = pd.read_csv("../lasso_training_data/feature_matrix_train.csv").drop(
    ["SampleID","Variant","Global CFR"], axis=1
).values
y_train = pd.read_csv("../lasso_training_data/feature_matrix_train.csv")["Global CFR"].values
X_test = pd.read_csv("../lasso_training_data/feature_matrix_test.csv").drop(
    ["SampleID","Variant","Global CFR"], axis=1
).values
y_test = pd.read_csv("../lasso_training_data/feature_matrix_test.csv")["Global CFR"].values

scaler = StandardScaler(with_mean=False)
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 1) Define a grid of alphas to try (log‐spaced from 1e-5 to 1e-2)
alphas = np.logspace(-5, -2, 30)

results = []
for a in alphas:
    l = Lasso(alpha=a, max_iter=10000)
    l.fit(X_train_s, y_train)
    n_feats = np.sum(l.coef_ != 0)
    r2 = r2_score(y_test, l.predict(X_test_s))
    results.append((a, n_feats, r2))

# 2) Put into a DataFrame for easy inspection
df = pd.DataFrame(results, columns=["alpha","n_features","test_R2"])
print(df)

# Elbow plot: number of features vs. test R^2 (used to select alpha for ML model)
plt.figure()
plt.plot(df["n_features"], df["test_R2"], marker='o')
plt.xlabel("Number of Features")
plt.ylabel("Test R\u00b2")
plt.title("Elbow Plot: n_features vs. Test R\u00b2")
plt.grid(True)
plt.show()