#!/usr/bin/env python3

"""
Script: bootstrap_model.py

Description:
    Performs parallelized bootstrap resampling to assess the stability of a Lasso model
    predicting global case fatality rate (CFR) from genomic features.

    Steps:
      1. Loads hardcoded train/test feature matrices (CSV filenames).
      2. Scales features (zero mean off).
      3. Runs N bootstrap iterations in parallel:
         a. Samples the train set with replacement.
         b. Fits Lasso(alpha) on the sample.
         c. Evaluates R² on the test set.
      4. Displays a histogram of bootstrap test-set R² distribution interactively.

Usage:
    python3 bootstrap_model.py

Dependencies:
    numpy, pandas, scikit-learn, matplotlib, joblib
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
import multiprocessing

# ---------------------------
# Configuration
# ---------------------------
TRAIN_CSV    = "feature_matrix_train.csv"
TEST_CSV     = "feature_matrix_test.csv"
ALPHA        = 0.000174    # Lasso regularization strength
N_BOOTSTRAPS = 1000        # Number of bootstrap iterations
SEED         = 42          # Random seed

# ---------------------------
# Data loading and scaling
# ---------------------------
def load_and_scale(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    feat_cols = df_train.columns.drop(["SampleID", "Variant", "Global CFR"])
    X_train = df_train[feat_cols].values
    y_train = df_train["Global CFR"].values
    X_test  = df_test[feat_cols].values
    y_test  = df_test["Global CFR"].values

    scaler = StandardScaler(with_mean=False)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    return X_train_s, y_train, X_test_s, y_test

# ---------------------------
# Single bootstrap function
# ---------------------------
def _bootstrap_iteration(i, X_train_s, y_train, X_test_s, y_test, alpha, seed):
    rng = np.random.RandomState(seed + i)
    idx = rng.choice(len(X_train_s), size=len(X_train_s), replace=True)
    X_bs, y_bs = X_train_s[idx], y_train[idx]
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_bs, y_bs)
    return r2_score(y_test, model.predict(X_test_s))

# ---------------------------
# Plotting function (interactive)
# ---------------------------
def show_histogram(r2_values):
    plt.figure(figsize=(8, 4))
    plt.hist(r2_values, bins=30, edgecolor='black')
    plt.xlabel("Test R²")
    plt.ylabel("Frequency")
    plt.title(f"Bootstrap Test R² Distribution (n={len(r2_values)})")
    plt.tight_layout()
    plt.show()

# ---------------------------
# Main execution
# ---------------------------
def main():
    print("Loading and scaling data...")
    X_train_s, y_train, X_test_s, y_test = load_and_scale(TRAIN_CSV, TEST_CSV)

    print(f"Running {N_BOOTSTRAPS} bootstrap iterations in parallel...")
    start_time = time.time()
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    r2_values = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_iteration)(i, X_train_s, y_train, X_test_s, y_test, ALPHA, SEED)
        for i in range(N_BOOTSTRAPS)
    )
    elapsed = time.time() - start_time
    print(f"Completed {N_BOOTSTRAPS} iterations in {elapsed:.1f} seconds.")

    show_histogram(r2_values)

if __name__ == '__main__':
    main()

