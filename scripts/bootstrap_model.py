#!/usr/bin/env python3

"""
Script: bootstrap_model.py

Description:
    Performs bootstrap resampling to assess the stability of a Lasso model predicting
    global case fatality rate (CFR) from genomic features and writes **only** the histogram
    of the bootstrap test-set R² distribution to `figures/bootstrap_r2_histogram.png`.

Usage:
    python3 bootstrap_model.py
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
import multiprocessing
import sys

# ---------------------------
# Configuration (hardcoded)
# ---------------------------
TRAIN_CSV    = "../lasso_training_data/feature_matrix_train.csv"
TEST_CSV     = "../lasso_training_data/feature_matrix_test.csv"
ALPHA        = 0.000174    # Lasso regularization strength
N_BOOTSTRAPS = 1000        # Number of bootstrap iterations
SEED         = 42          # Base random seed

def get_output_dir():
    this_path = os.path.abspath(__file__)
    parent = os.path.dirname(this_path)
    if os.path.basename(parent) == "scripts":
        project_root = os.path.dirname(parent)
    else:
        project_root = parent
    out_dir = os.path.join(project_root, "figures")
    return os.path.abspath(out_dir)

OUT_DIR = get_output_dir()

# ---------------------------
# Load and scale
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
# Single bootstrap iteration
# ---------------------------
def _bootstrap_iteration(i, X_train_s, y_train, X_test_s, y_test, alpha, seed):
    rng = np.random.RandomState(seed + i)
    idx = rng.choice(len(X_train_s), size=len(X_train_s), replace=True)
    X_bs, y_bs = X_train_s[idx], y_train[idx]
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_bs, y_bs)
    return r2_score(y_test, model.predict(X_test_s))

# ---------------------------
# Plot histogram only
# ---------------------------
def plot_histogram(r2_values, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    arr = np.array(r2_values)
    hist_path = os.path.join(out_dir, "bootstrap_r2_histogram.png")

    plt.figure(figsize=(8, 4))
    plt.hist(arr, bins=30, edgecolor='black')
    plt.xlabel("Test R²")
    plt.ylabel("Frequency")
    plt.title(f"Bootstrap Test R² Distribution (n={len(arr)})")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Saved histogram to {hist_path}")

# ---------------------------
# Main
# ---------------------------
def main():
    print("Loading and scaling data...")
    X_train_s, y_train, X_test_s, y_test = load_and_scale(TRAIN_CSV, TEST_CSV)

    print(f"Running {N_BOOTSTRAPS} bootstrap iterations...")
    start = time.time()
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    r2_values = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_iteration)(i, X_train_s, y_train, X_test_s, y_test, ALPHA, SEED)
        for i in range(N_BOOTSTRAPS)
    )
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s.")

    plot_histogram(r2_values, OUT_DIR)

if __name__ == "__main__":
    main()
