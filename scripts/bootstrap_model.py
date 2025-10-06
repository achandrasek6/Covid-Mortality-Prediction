#!/usr/bin/env python3

"""
Script: bootstrap_model.py

Description:
    Performs bootstrap resampling to assess the stability of a Lasso model predicting
    global case fatality rate (CFR) from genomic features and writes **only** a smooth
    KDE "bell-curve" of the bootstrap test-set R² distribution to
    `figures/bootstrap_r2_histogram.png`. The plot shades the central 95% interval
    in green, the tails in red, and shows a vertical dotted line at the model R² = 0.8306.

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
from sklearn.neighbors import KernelDensity
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
MODEL_R2     = 0.8306      # Annotated vertical line

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

    # with_mean=False to be CSR/COO-friendly if sparse; safe for dense too.
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
# Plot smooth KDE with CI shading
# ---------------------------
def plot_density_with_ci(r2_values, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    arr = np.asarray(r2_values, dtype=float)
    out_path = os.path.join(out_dir, "bootstrap_r2_histogram.png")  # keep same name to replace

    # KDE grid
    x_min, x_max = np.min(arr), np.max(arr)
    pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    x_grid = np.linspace(x_min - pad, x_max + pad, 1000)[:, None]

    # Bandwidth via Silverman's rule-of-thumb (robust-ish)
    std = np.std(arr, ddof=1) if len(arr) > 1 else 0.1
    n = len(arr)
    bw = 1.06 * std * (n ** (-1/5)) if n > 1 and std > 0 else 0.1

    kde = KernelDensity(kernel="gaussian", bandwidth=bw)
    kde.fit(arr[:, None])
    log_dens = kde.score_samples(x_grid)
    y = np.exp(log_dens)  # density

    # Empirical 95% interval from percentiles
    lo, hi = np.percentile(arr, [2.5, 97.5])

    # Build masks for shading
    x = x_grid.ravel()
    inside = (x >= lo) & (x <= hi)
    left_tail = x < lo
    right_tail = x > hi

    plt.figure(figsize=(9, 4.8))

    # Plot the smooth density curve
    plt.plot(x, y, linewidth=2)

    # Shade tails (red) and central 95% (green)
    # Left tail
    if np.any(left_tail):
        plt.fill_between(x[left_tail], y[left_tail], 0, alpha=0.25, color="red")
    # Central 95%
    if np.any(inside):
        plt.fill_between(x[inside], y[inside], 0, alpha=0.25, color="green")
    # Right tail
    if np.any(right_tail):
        plt.fill_between(x[right_tail], y[right_tail], 0, alpha=0.25, color="red")

    # Draw vertical dotted line at the model R² with label
    plt.axvline(MODEL_R2, linestyle=":", linewidth=2)
    # Place a small label near the top of the curve at that x
    # Find y-value at MODEL_R2 (nearest point)
    idx_near = np.argmin(np.abs(x - MODEL_R2))
    y_near = y[idx_near]
    plt.text(MODEL_R2, y_near, f"  Model R² = {MODEL_R2:.4f}", rotation=90, va="bottom", ha="left")

    # Axes/labels
    plt.xlabel("Test R²")
    plt.ylabel("Density")
    plt.title(f"Bootstrap Test R² Density (n={len(arr)}), 95% CI [{lo:.3f}, {hi:.3f}]")

    # Nice margins
    plt.margins(x=0.02, y=0.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved density plot to {out_path}")

# ---------------------------
# Main
# ---------------------------
def main():
    print("Loading and scaling data...")
    X_train_s, y_train, X_test_s, y_test = load_and_scale(TRAIN_CSV, TEST_CSV)

    print(f"Running {N_BOOTSTRAPS} bootstrap iterations...")
    start = time.time()
    n_jobs = max(1, multiprocessing.cpu_count() - 1)  # keep cores parallelization
    r2_values = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_iteration)(i, X_train_s, y_train, X_test_s, y_test, ALPHA, SEED)
        for i in range(N_BOOTSTRAPS)
    )
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s.")

    plot_density_with_ci(r2_values, OUT_DIR)

if __name__ == "__main__":
    main()
