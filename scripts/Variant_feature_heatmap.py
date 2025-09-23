#!/usr/bin/env python3
"""
variant_feature_heatmap.py

- Concatenate train+test; exclude WildType.
- Group features into ORF1ab / Spike (S) / Other.
- Draw a REGION HEADER STRIP above the heatmap using exact block boundaries.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---- Paths ----
TRAIN_CSV = "../lasso_training_data/feature_matrix_train.csv"
TEST_CSV  = "../lasso_training_data/feature_matrix_test.csv"
SAVE_PNG  = "../figures/variant_feature_heatmap.png"

# ---- Config ----
NON_FEATURE_COLS = {"SampleID", "Variant", "Global CFR"}
EXCLUDE_VARIANTS = {"WildType"}
PREFERRED_ORDER  = ["Alpha", "Beta", "Gamma", "Delta", "Omicron"]
SHOW_DEBUG = False  # set True to print region counts & first few col names

# Colors for the header strip (neutral vs. viridis)
HEADER_COLORS = {
    "ORF1ab": "#90EE90",  # light blue
    "S":      "#F28E2B",  # orange
    "Misc":   "#E15759",  # soft red
}

DISPLAY_LABELS = {
    "ORF1ab": "ORF1ab (replicase)",
    "S":      "Spike (S)",
    "Misc":   "Other (N/M/E etc.)",
}

def feature_region(col: str) -> str:
    # Be forgiving about prefixes/casing used in your CSVs
    c = col.strip()
    if c.startswith(("ORF1ab_", "ORF1_", "orf1ab_", "orf1_")):
        return "ORF1ab"
    if c.startswith(("S_", "s_")):
        return "S"
    return "Misc"

# ---- Load & combine ----
dfs = []
for p in (TRAIN_CSV, TEST_CSV):
    if os.path.exists(p):
        dfs.append(pd.read_csv(p))
    else:
        print(f"[warn] missing: {p}")
if not dfs:
    raise FileNotFoundError("No input CSVs found.")
df = pd.concat(dfs, ignore_index=True)

# ---- Filter & pick features ----
if "Variant" not in df.columns:
    raise KeyError("Column 'Variant' missing.")
df = df[~df["Variant"].isin(EXCLUDE_VARIANTS)]

feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
if not feature_cols:
    raise ValueError("No feature columns found.")

# ---- Aggregate ----
vm = df.groupby("Variant")[feature_cols].mean()

# Row order
order = [v for v in PREFERRED_ORDER if v in vm.index]
order += [v for v in vm.index if v not in order]
vm = vm.reindex(order)

# ---- Column grouping & precise blocks ----
# First, keep original column order from the combined DF
regions_map = {c: feature_region(c) for c in feature_cols}

# Then reorder columns by region while PRESERVING within-region order
ordered_cols = (
    [c for c in feature_cols if regions_map[c] == "ORF1ab"] +
    [c for c in feature_cols if regions_map[c] == "S"] +
    [c for c in feature_cols if regions_map[c] == "Misc"]
)
vm = vm[ordered_cols]

# A vector of region labels aligned 1:1 with ordered columns
col_regions = np.array([regions_map[c] for c in ordered_cols])

# Find exact change points (indices where region label switches)
change_ix = np.where(col_regions[1:] != col_regions[:-1])[0] + 1
bounds = np.concatenate(([0], change_ix, [len(col_regions)]))  # e.g., [0, 512, 640, 700]
centers = (bounds[:-1] + bounds[1:]) / 2.0
region_blocks = [col_regions[bounds[i]] for i in range(len(bounds) - 1)]  # names per block
region_names_unique = region_blocks  # already in display order

if SHOW_DEBUG:
    from collections import Counter
    print("[debug] region counts:", Counter(col_regions))
    print("[debug] bounds:", bounds.tolist())
    print("[debug] first 5 cols:", ordered_cols[:5])

# ---- Header strip colormap aligned to blocks ----
# We color per feature, but you’ll also get clear block labels from bounds.
palette = {"ORF1ab": HEADER_COLORS["ORF1ab"],
           "S":      HEADER_COLORS["S"],
           "Misc":   HEADER_COLORS["Misc"]}
header_color_row = np.array([palette[r] for r in col_regions], dtype=object)

# ---- Plot ----
variants = vm.index.tolist()
mat = vm.values
w = min(18, max(10, mat.shape[1] / 80.0 + 10))
h_main = 3 + len(variants) * 0.5
h_header = 0.70

fig = plt.figure(figsize=(w, h_main + h_header + 0.6), constrained_layout=True)
gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[h_header, h_main], hspace=0.0)
ax_header = fig.add_subplot(gs[0, 0])
ax = fig.add_subplot(gs[1, 0])

# Main heatmap
im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis")
cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label("Fraction of samples with feature = 1", labelpad=10, fontsize=12)

# Y-axis
ax.set_yticks(np.arange(len(variants)))
ax.set_yticklabels(variants, fontsize=14)

# X-axis ticks at block centers with explicit labels
ax.set_xticks(centers)
ax.set_xticklabels([DISPLAY_LABELS.get(name, name) for name in region_blocks], fontsize=14)
ax.set_xlabel("Feature regions", fontsize=16, fontweight="bold")
ax.set_ylabel("Variant", fontsize=16, fontweight="bold")


# ---- Header strip (exact alignment) ----
# Draw colored rectangles per block to avoid any rounding issues
ax_header.set_xlim(-0.5, len(col_regions) - 0.5)
ax_header.set_ylim(0, 1)
ax_header.axis("off")

for i in range(len(bounds) - 1):
    left = bounds[i]   - 0.5
    right = bounds[i+1] - 0.5
    name = region_blocks[i]
    ax_header.axvspan(left, right, facecolor=HEADER_COLORS[name], ec="none")

    # Centered block label (short form on header)
    ax_header.text((left + right) / 2.0, 0.5, name, ha="center", va="center", fontsize=14, fontweight="bold")

ax_header.set_title(
    "Variant × Feature Presence (by Region)",
    pad=6, fontsize=18, fontweight="bold"
)

os.makedirs(os.path.dirname(SAVE_PNG), exist_ok=True)
plt.savefig(SAVE_PNG, dpi=300)
print(f"[ok] Saved heatmap to: {SAVE_PNG}")
