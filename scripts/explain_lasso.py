#!/usr/bin/env python3
"""
explain_lasso.py

Evaluate and explain a trained Lasso regression model (saved earlier by your training script)
using SHAP (global + per-feature attributions) and LIME (local, per-sample explanations).

Outputs:
  - SHAP:
      * <outdir>/shap_summary_bar.png      (global feature importance: mean |SHAP value|)
      * <outdir>/shap_beeswarm.png         (per-feature distribution of SHAP values)
      * <outdir>/shap_importances.csv      (table of features sorted by mean |SHAP|)
  - LIME:
      * <outdir>/lime_sample_<i>.html      (interactive HTML explanations for selected samples)
      * <outdir>/lime_weights_sample_<i>.csv  (top K weighted features for each explained sample)
  - Diagnostics:
      * <outdir>/metrics.txt               (R^2 and MSE on test set, plus a short summary)

Assumptions:
  - Your feature CSVs include columns: 'SampleID', 'Variant', and 'Global CFR' (target).
  - All other columns are binary/continuous features.
  - A StandardScaler and Lasso model were persisted with joblib or pickle under --artifacts_dir.

Install (if needed):
  pip install shap lime scikit-learn pandas numpy joblib matplotlib

Usage (defaults try to match your existing layout):
  python3 scripts/explain_lasso.py \
  --train_csv lasso_training_data/feature_matrix_train.csv \
  --test_csv  lasso_training_data/feature_matrix_test.csv \
  --artifacts_dir model_artifacts \
  --outdir explanations \
  --lime_n 5 \
  --lime_select largest_error \
  --lime_space raw \
  --lime_digits 6

Notes:
  - If artifact filenames are unknown, the script will attempt to auto-detect the scaler and model
    inside --artifacts_dir by filename patterns ('scaler', 'model', 'lasso'). You can override via
    --scaler_path and --model_path explicitly.
"""

import argparse
import os
import sys
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

# Optional libraries: handled with a friendly error if missing
try:
    import shap
except Exception as e:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None


TARGET_COL = "Global CFR"
META_COLS = {"SampleID", "Variant", TARGET_COL}


def _find_artifact(artifacts_dir: Path, kind: str) -> Optional[Path]:
    """
    Try to locate an artifact file by heuristic patterns.
    kind in {'scaler', 'model'}.
    """
    patterns = []
    if kind == "scaler":
        patterns = ["*scaler*.joblib", "*scaler*.pkl"]
    elif kind == "model":
        patterns = ["*lasso*model*.joblib", "*lasso*model*.pkl", "*model*.joblib", "*model*.pkl"]
    else:
        return None
    cands = []
    for pat in patterns:
        cands += list(artifacts_dir.glob(pat))
    if not cands:
        return None
    # Pick the most recently modified candidate
    cands = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def load_artifacts(artifacts_dir: Path,
                   scaler_path: Optional[Path],
                   model_path: Optional[Path]) -> Tuple[StandardScaler, Lasso]:
    """
    Load the persisted StandardScaler and Lasso model. If explicit paths are not provided,
    auto-detect files within artifacts_dir.
    """
    if scaler_path is None:
        scaler_path = _find_artifact(artifacts_dir, "scaler")
    if model_path is None:
        model_path = _find_artifact(artifacts_dir, "model")

    if scaler_path is None or not scaler_path.exists():
        raise FileNotFoundError("Could not locate scaler artifact. Provide --scaler_path.")
    if model_path is None or not model_path.exists():
        raise FileNotFoundError("Could not locate model artifact. Provide --model_path.")

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return scaler, model


def load_split(train_csv: Path, test_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train/test CSVs and enforce numeric target column.
    """
    df_train = pd.read_csv(train_csv)
    df_test  = pd.read_csv(test_csv)

    # Coerce to numeric target
    for df in (df_train, df_test):
        if TARGET_COL in df.columns:
            df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')

    # Drop rows with missing target
    df_train = df_train.dropna(subset=[TARGET_COL])
    df_test  = df_test.dropna(subset=[TARGET_COL])

    return df_train, df_test


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str], Optional[pd.Series]]:
    """
    Returns (X, y, feature_names, sample_ids) for a feature matrix dataframe.
    """
    cols = [c for c in df.columns if c not in META_COLS]
    X = df[cols].astype(float)  # features may be binary or continuous
    y = df[TARGET_COL].to_numpy(dtype=float)
    sample_ids = df["SampleID"] if "SampleID" in df.columns else None
    return X, y, cols, sample_ids


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mse


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_shap_summary(shap_values: np.ndarray, feature_names: List[str], outdir: Path) -> None:
    # Bar plot (global mean |SHAP|)
    plt.figure()
    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(outdir / "shap_summary_bar.png", dpi=200)
    plt.close()

    # Beeswarm plot (distribution)
    plt.figure()
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(outdir / "shap_beeswarm.png", dpi=200)
    plt.close()


def save_shap_importances(shap_values: np.ndarray, feature_names: List[str], outpath: Path) -> pd.DataFrame:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    df_imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df_imp = df_imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df_imp.to_csv(outpath, index=False)
    return df_imp


def build_lime_explainer(X_train: np.ndarray, feature_names: List[str], class_names: List[str]) -> LimeTabularExplainer:
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="regression",
        discretize_continuous=True,
        verbose=False,
        random_state=42,
    )
    return explainer


def explain_with_lime(explainer: LimeTabularExplainer,
                      instance: np.ndarray,
                      predict_fn,
                      feature_names: List[str],
                      outdir: Path,
                      sample_name: str,
                      top_features: int = 15):
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_fn,
        num_features=min(top_features, len(feature_names))
    )
    # Save HTML
    html_path = outdir / f"lime_{sample_name}.html"
    try:
        exp.save_to_file(str(html_path))
    except Exception:
        # Fallback: write as HTML string
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(exp.as_html())

    # Save weights to CSV
    weights = exp.as_list()
    dfw = pd.DataFrame(weights, columns=["feature", "weight"])
    dfw.to_csv(outdir / f"lime_weights_{sample_name}.csv", index=False)


def render_lime_table(exp,
                      instance_raw: np.ndarray,
                      feature_names: List[str],
                      outpath: Path,
                      digits: int = 6) -> None:
    """
    Build a high-precision HTML table for a LIME explanation using the raw (unscaled) feature values.
    """
    # LIME (regression) exposes a single label; get (feature_idx, weight) pairs
    label = next(iter(exp.as_map().keys()))
    rows = []
    for feat_idx, weight in exp.as_map()[label]:
        rows.append({
            "Feature": feature_names[feat_idx],
            "Value": float(instance_raw[feat_idx]),
            "Weight": float(weight),
        })
    df = pd.DataFrame(rows).sort_values("Weight", key=lambda s: s.abs(), ascending=False)
    html = (df.style
              .format({"Value": f"{{:.{digits}f}}", "Weight": f"{{:.{digits}f}}"})
              .hide(axis="index")
              .to_html())
    Path(outpath).write_text(html, encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Explain a trained Lasso model using SHAP & LIME.")
    parser.add_argument("--train_csv", type=Path, default=Path("../lasso_training_data/feature_matrix_train.csv"))
    parser.add_argument("--test_csv",  type=Path, default=Path("../lasso_training_data/feature_matrix_test.csv"))
    parser.add_argument("--artifacts_dir", type=Path, default=Path("../model_artifacts"))
    parser.add_argument("--scaler_path", type=Path, default=None, help="Optional explicit path to scaler artifact.")
    parser.add_argument("--model_path",  type=Path, default=None, help="Optional explicit path to lasso model artifact.")
    parser.add_argument("--outdir", type=Path, default=Path("explanations"))
    parser.add_argument("--max_shap_samples", type=int, default=2000, help="Cap number of rows used for SHAP to keep plotting responsive.")
    parser.add_argument("--lime_n", type=int, default=5, help="How many samples to explain with LIME.")
    parser.add_argument("--lime_select", type=str, choices=["largest_error", "random", "highest_pred"], default="largest_error",
                        help="Strategy for picking LIME samples from test set.")
    parser.add_argument("--lime_top_features", type=int, default=15)
    parser.add_argument("--lime_digits", type=int, default=6, help="Decimals for LIME HTML table.")
    parser.add_argument("--lime_space", choices=["raw", "scaled"], default="raw",
                        help="Explain in raw feature space (recommended); scale inside predict_fn.")
    args = parser.parse_args()

    ensure_outdir(args.outdir)

    # Load data
    df_train, df_test = load_split(args.train_csv, args.test_csv)
    X_train_df, y_train, feature_names, train_ids = split_features_target(df_train)
    X_test_df,  y_test,  _,         test_ids  = split_features_target(df_test)

    # Load artifacts
    scaler, model = load_artifacts(args.artifacts_dir, args.scaler_path, args.model_path)

    # Scale features (must match training-time behavior)
    X_train = scaler.transform(X_train_df.values)
    X_test  = scaler.transform(X_test_df.values)

    # Predictions & metrics
    y_pred = model.predict(X_test)
    r2, mse = compute_metrics(y_test, y_pred)

    # Write metrics to disk
    with open(args.outdir / "metrics.txt", "w") as f:
        f.write(f"Test R^2: {r2:.4f}\n")
        f.write(f"Test MSE: {mse:.6e}\n")
        if test_ids is not None:
            # top residuals
            resid = np.abs(y_test - y_pred)
            order = np.argsort(resid)[::-1][:10]
            f.write("\nTop 10 absolute residuals (SampleID, y_true, y_pred, |err|):\n")
            for idx in order:
                sid = str(test_ids.iloc[idx]) if test_ids is not None else f"idx_{idx}"
                f.write(f"{sid}, {y_test[idx]:.6f}, {y_pred[idx]:.6f}, {resid[idx]:.6f}\n")

    # ====== SHAP ======
    if shap is None:
        print("WARNING: 'shap' is not installed. Skipping SHAP plots.", file=sys.stderr)
    else:
        try:

            import matplotlib.pyplot as plt
            from matplotlib import colors as mcolors
            from matplotlib.lines import Line2D

            # Linear model on *scaled* features
            explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")

            # sample MANY rows (random, not a [:n] slice)
            n = min(args.max_shap_samples, X_test.shape[0])  # e.g., --max_shap_samples 500
            X_shap = shap.utils.sample(X_test, n, random_state=42)

            shap_values = explainer.shap_values(X_shap)
            sv = np.asarray(shap_values)  # (n_samples, n_features)

            # ---- Density strip (no jitter; alpha ~ local density) ----
            max_display = 20
            bins = 50
            min_alpha, max_alpha = 0.06, 0.90

            mean_abs = np.mean(np.abs(sv), axis=0)
            top_idx = np.argsort(mean_abs)[-max_display:][::-1]
            feat_names = np.asarray(feature_names)
            cmap = plt.get_cmap("coolwarm")

            # If you have RAW (unscaled) features aligned to X_shap, use them for color:
            # X_color = X_test_df.values
            X_color = X_shap  # fallback: uses the same matrix

            fig, ax = plt.subplots(figsize=(8, 10))

            # plot rows with PER-FEATURE coloring (robust) to avoid "all blue"
            for row, j in enumerate(top_idx):
                xvals = sv[:, j]  # SHAP values for feature j across samples
                fvals = X_color[:, j]  # values for color on this row

                # Decide coloring mode: binary vs continuous (robust per-feature norm)
                valid = fvals[~np.isnan(fvals)]
                uniq = np.unique(valid)
                is_binary = uniq.size <= 3 and set(np.round(uniq, 6)).issubset({0, 1})

                if is_binary:
                    # two fixed colors for 0/1
                    rgba = np.zeros((len(fvals), 4))
                    rgba[fvals > 0.5] = (0.79, 0.19, 0.19, 1.0)  # red-ish (high)
                    rgba[fvals <= 0.5] = (0.18, 0.40, 0.77, 1.0)  # blue-ish (low)
                else:
                    # robust per-feature normalization (5th–95th percentile)
                    lo = np.nanpercentile(fvals, 5)
                    hi = np.nanpercentile(fvals, 95)
                    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                        lo, hi = np.nanmin(fvals), np.nanmax(fvals)
                        if not np.isfinite(lo) or lo == hi:
                            lo, hi = -1.0, 1.0  # last resort
                    norm = mcolors.Normalize(vmin=lo, vmax=hi)
                    rgba = cmap(norm(fvals))

                # alpha by 1D histogram density on the SHAP axis
                counts, edges = np.histogram(xvals, bins=bins)
                bidx = np.digitize(xvals, edges) - 1
                bidx = np.clip(bidx, 0, len(counts) - 1)
                dens = counts[bidx].astype(float)
                if dens.max() > 0:
                    dens = (dens - dens.min()) / (dens.max() - dens.min())
                alpha = min_alpha + (max_alpha - min_alpha) * dens
                rgba[:, 3] = alpha

                # plot all points at the same y (no vertical jitter)
                y = np.full_like(xvals, row, dtype=float)
                ax.scatter(xvals, y, c=rgba, s=18, linewidths=0)

            ax.set_yticks(range(len(top_idx)))
            ax.set_yticklabels(feat_names[top_idx])
            ax.set_xlabel("SHAP value (impact on model output)")
            ax.set_ylim(-0.5, len(top_idx) - 0.5)
            ax.axvline(0.0, color="grey", lw=1)

            # Legends: opacity (density) + color meaning
            alpha_handles = [
                Line2D([0], [0], marker='o', color='none', label='Low density',
                       markerfacecolor=(0.3, 0.5, 0.9, min_alpha), markersize=8),
                Line2D([0], [0], marker='o', color='none', label='High density',
                       markerfacecolor=(0.3, 0.5, 0.9, max_alpha), markersize=8),
            ]
            color_handles = [
                Line2D([0], [0], marker='o', color='none',
                       markerfacecolor=(0.18, 0.40, 0.77, 1.0), label='Reference / Absent', markersize=8),
                Line2D([0], [0], marker='o', color='none',
                       markerfacecolor=(0.79, 0.19, 0.19, 1.0), label='Variant present', markersize=8),
            ]
            ax.legend(handles=color_handles + alpha_handles, title="Encoding", loc="upper right", frameon=False)

            fig.tight_layout()
            fig.savefig(args.outdir / "shap_strip_density.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

            # importances CSV (unchanged)
            save_shap_importances(sv, feature_names, args.outdir / "shap_importances.csv")

        except Exception as e:
            print(f"WARNING: SHAP failed: {e}", file=sys.stderr)

    # --- Global importance bar (Top 10) ---

    import matplotlib.pyplot as plt
    import pandas as pd

    top_k = 10
    mean_abs = np.mean(np.abs(sv), axis=0)  # mean(|SHAP|) per feature
    order = np.argsort(mean_abs)[-top_k:][::-1]  # indices of top-k, descending
    feat_top = np.asarray(feature_names)[order]
    vals_top = mean_abs[order]

    # plot
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.barh(feat_top[::-1], vals_top[::-1])  # reverse for top at bottom
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_ylabel("Feature")
    # annotate values at bar ends
    for i, v in enumerate(vals_top[::-1]):
        ax.text(v, i, f" {v:.4g}", va="center")

    fig.tight_layout()
    fig.savefig(args.outdir / "shap_top10_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # save CSV
    pd.DataFrame({"feature": feat_top, "mean_abs_shap": vals_top}) \
        .to_csv(args.outdir / "shap_top10.csv", index=False)

    #== LIME ===
    if LimeTabularExplainer is None:
        print("WARNING: 'lime' is not installed. Skipping LIME explanations.", file=sys.stderr)
    else:
        try:
            # Choose data space for LIME
            if args.lime_space == "raw":
                training_for_lime = X_train_df.values  # raw features
                instance_matrix = X_test_df.values  # raw features

                def predict_fn(arr: np.ndarray) -> np.ndarray:  # scale inside
                    return model.predict(scaler.transform(arr))
            else:
                training_for_lime = X_train  # scaled features
                instance_matrix = X_test  # scaled features

                def predict_fn(arr: np.ndarray) -> np.ndarray:
                    return model.predict(arr)

            explainer = LimeTabularExplainer(
                training_data=training_for_lime,
                feature_names=feature_names,
                class_names=["Global CFR"],
                mode="regression",
                discretize_continuous=False,  # << turn off binning / low-precision thresholds
                verbose=False,
                random_state=42,
            )

            # Pick samples to explain
            n = min(args.lime_n, len(X_test))
            if args.lime_select == "random":
                idxs = np.random.RandomState(42).choice(len(X_test), size=n, replace=False)
            elif args.lime_select == "highest_pred":
                idxs = np.argsort(y_pred)[-n:][::-1]
            else:
                errors = np.abs(y_test - y_pred)
                idxs = np.argsort(errors)[-n:][::-1]

            for rank, i in enumerate(idxs):
                sid = str(test_ids.iloc[i]) if test_ids is not None else f"row_{i}"
                sample_name = f"sample_{rank + 1}_{sid}"

                # Explain
                exp = explainer.explain_instance(
                    data_row=instance_matrix[i],
                    predict_fn=predict_fn,
                    num_features=min(args.lime_top_features, len(feature_names)),
                )

                # 1) Keep LIME's original HTML (bars)
                html_path = args.outdir / f"lime_{sample_name}.html"
                try:
                    exp.save_to_file(str(html_path))
                except Exception:
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(exp.as_html())

                # 2) High-precision table with raw values + weights
                if args.lime_space == "raw":
                    raw_row = instance_matrix[i]
                else:
                    # invert scaling to show raw values
                    raw_row = scaler.inverse_transform(instance_matrix[i].reshape(1, -1))[0]

                render_lime_table(
                    exp=exp,
                    instance_raw=raw_row,
                    feature_names=feature_names,
                    outpath=args.outdir / f"lime_table_{sample_name}.html",
                    digits=args.lime_digits,
                )

                # CSV with weights (unchanged)
                weights = exp.as_list()
                pd.DataFrame(weights, columns=["feature", "weight"]) \
                    .to_csv(args.outdir / f"lime_weights_{sample_name}.csv", index=False)

        except Exception as e:
            print(f"WARNING: LIME failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
