#!/usr/bin/env python3
"""
Script: variant_feature_heatmap.py

Description:
    Reads in a binary feature matrix CSV (feature_matrix_train.csv), groups samples
    by their pangolin lineage (“Variant”), computes the fraction of positive calls
    per feature within each variant, and then displays a heatmap of Variant × Feature.

Usage:
    python variant_feature_heatmap.py

Dependencies:
    - Python 3
    - pandas
    - matplotlib
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
csv_file = "../lasso_training_data/feature_matrix_train.csv"
df = pd.read_csv(csv_file)

# Select only the binary features
feature_cols = df.columns.drop(["SampleID", "Variant", "Global CFR"])

# Compute per-variant mean (i.e. fraction of samples with value=1 for each feature)
variant_matrix = df.groupby("Variant")[feature_cols].mean()

# Enforce order and fill any missing variant with 0’s
ordered = ["Alpha", "Beta", "Gamma", "Delta", "Omicron", "WildType"]
variant_matrix = variant_matrix.reindex(ordered, fill_value=0)

# Prepare for plotting
variants = variant_matrix.index.tolist()
features = variant_matrix.columns.tolist()
matrix = variant_matrix.values  # shape: (n_variants, n_features)

# Plot
fig, ax = plt.subplots(figsize=(14, 3 + len(variants) * 0.5))
cax = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")

# Colorbar
cbar = fig.colorbar(cax, ax=ax, pad=0.02)
cbar.set_label("Fraction of samples with feature = 1", labelpad=10)

# Axis ticks
ax.set_yticks(np.arange(len(variants)))
ax.set_yticklabels(variants, fontsize=10)
# Show every nth feature label to avoid overlap
step = max(1, len(features) // 50)
ax.set_xticks(np.arange(0, len(features), step))
ax.set_xticklabels(features[::step], rotation=90, fontsize=6)

ax.set_xlabel("Features")
ax.set_ylabel("Variant")
ax.set_title("Per-Variant Presence Fraction Heatmap")
plt.tight_layout()
plt.show()