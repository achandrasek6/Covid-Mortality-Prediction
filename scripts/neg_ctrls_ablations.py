#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run negative controls and ablations for a regression model on genomic feature
matrices. **Python 3** version (uses pathlib, type hints, f-strings).

Features:
  • Baseline training/eval (StandardScaler + Lasso or LassoCV)
  • Label permutation controls (train on shuffled y)
  • Feature-column shuffles (shuffle each feature column across rows in TRAIN only)
  • Ablations (drop features by regex/prefix list or from a file)
  • Optional ablation of top-k features ranked by |coef| from baseline
  • Plots + CSVs for all experiments

Example:
  python3 scripts/neg_ctrls_ablations.py \
    --train_csv lasso_training_data/feature_matrix_train.csv \
    --test_csv  lasso_training_data/feature_matrix_test.csv  \
    --target_col cfr \
    --id_col sample_id \
    --outdir controls_out \
    --use_lassocv --cv_folds 5 \
    --n_label_perm 200 \
    --n_feat_shuffle 100 \
    --ablate_regex "^S_" "^ORF1ab_" \
    --ablate_list key_sites.txt \
    --ablate_topk_coef 50 \
    --save_preds
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Optional: Spearman (SciPy) with numpy fallback
try:
    from scipy.stats import spearmanr  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ------------------------ Utilities ------------------------

def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if _HAVE_SCIPY:
        return float(spearmanr(y_true, y_pred, nan_policy="omit").correlation)

    # Numpy fallback (average ranks; ties handled approximately)
    def rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a))
        return ranks

    r1, r2 = rank(y_true), rank(y_pred)
    r1m, r2m = r1 - r1.mean(), r2 - r2.mean()
    denom = np.sqrt((r1m ** 2).sum() * (r2m ** 2).sum()) or np.nan
    return float((r1m @ r2m) / denom)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(obj: Dict, path: Path) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False)


# ------------------------ Argparse ------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Negative controls + ablations runner (Py3)")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--target_col", default="target")
    p.add_argument("--id_col", default=None)
    p.add_argument("--outdir", required=True)

    # Model
    p.add_argument("--alpha", type=float, default=0.01, help="Lasso alpha (ignored if use_lassocv)")
    p.add_argument("--use_lassocv", action="store_true", help="Use LassoCV to pick alpha")
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--random_state", type=int, default=42)

    # Negative controls
    p.add_argument("--n_label_perm", type=int, default=100)
    p.add_argument("--n_feat_shuffle", type=int, default=50)

    # Ablations
    p.add_argument("--ablate_regex", nargs="*", default=[], help="Regex patterns of features to drop (e.g., ^S_ ^ORF1ab_)")
    p.add_argument("--ablate_list", default=None, help="File with one feature per line to drop")
    p.add_argument("--ablate_topk_coef", type=int, default=0, help="Drop top-k features by |coef| from baseline model")

    # Misc
    p.add_argument("--save_preds", action="store_true")

    # Utilities
    p.add_argument("--list_columns", action="store_true", help="Print columns of train/test CSVs and exit")

    return p.parse_args()


# ------------------------ IO ------------------------

def load_matrix(path: str | Path, target_col: str, id_col: Optional[str]) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """Load matrix and resolve the target column robustly.

    Matching logic:
      1) Exact match
      2) Case-insensitive exact match
      3) Normalized match (strip non-alnum, lowercase), e.g., "cfr" → "Global CFR"
      4) Normalized substring match if unique (e.g., "fatality" → "Case Fatality Rate")
    """
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    path = Path(path)
    df = pd.read_csv(path)

    # Resolve target column
    orig_cols = list(df.columns)
    if target_col in df.columns:
        resolved = target_col
    else:
        # Case-insensitive map
        lower_map = {c.lower(): c for c in df.columns}
        t_lo = (target_col or "").lower()
        if t_lo in lower_map:
            resolved = lower_map[t_lo]
        else:
            # Normalized match (remove spaces/punct)
            norm_map = {_norm(c): c for c in df.columns}
            t_norm = _norm(target_col or "")
            if t_norm in norm_map:
                resolved = norm_map[t_norm]
            else:
                # Unique substring match on normalized names
                cand = [c for n, c in norm_map.items() if t_norm and t_norm in n]
                cand = sorted(set(cand))
                if len(cand) == 1:
                    resolved = cand[0]
                else:
                    preview = ", ".join(orig_cols[:20]) + (" ..." if len(orig_cols) > 20 else "")
                    details = (f"\nCandidates (normalized contains match): {cand}" if cand else "")
                    raise ValueError(
                        f"Target column '{target_col}' not found in {path}. Available columns (first 20): {preview}{details}"
                    )
    y = df[resolved].astype(float)

    id_series = df[id_col] if id_col and (id_col in df.columns) else None
    drop_cols = [c for c in [resolved, id_col] if c in df.columns]
    X = df.drop(columns=drop_cols)
    # enforce numeric
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X, y, id_series


# ------------------------ Modeling ------------------------

def build_model(args: argparse.Namespace) -> Pipeline:
    if args.use_lassocv:
        model = LassoCV(cv=args.cv_folds, random_state=args.random_state, n_alphas=100, max_iter=10000)
    else:
        model = Lasso(alpha=args.alpha, random_state=args.random_state, max_iter=10000)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", model),
    ])
    return pipe


def fit_eval(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
             X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Dict[str, float], np.ndarray]:
    pipe.fit(X_train.values, y_train.values)
    y_pred = pipe.predict(X_test.values)
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "spearman": float(spearman_corr(y_test.values, y_pred)),
    }
    return metrics, y_pred


# ------------------------ Plots ------------------------

def plot_hist(values: np.ndarray, baseline: float, xlabel: str, title: str, outpath: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(values, bins=30)
    ax.axvline(baseline, color="red", linestyle="--", label=f"baseline = {baseline:.4f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------ Experiments ------------------------

def collect_features_to_drop(X_cols: List[str], regex_list: List[str], list_file: Optional[str],
                             topk_by_coef: int, baseline_pipe: Pipeline) -> List[Tuple[str, List[str]]]:
    drops: List[Tuple[str, List[str]]] = []

    # regex-based
    if regex_list:
        for pat in regex_list:
            rx = re.compile(pat)
            cols = [c for c in X_cols if rx.search(c)]
            if cols:
                drops.append((f"regex:{pat}", cols))

    # list-file
    if list_file:
        lst = [ln.strip() for ln in Path(list_file).read_text().splitlines() if ln.strip()]
        cols = [c for c in X_cols if c in lst]
        if cols:
            drops.append((f"list:{Path(list_file).name}", cols))

    # top-k by |coef|
    if topk_by_coef and hasattr(baseline_pipe.named_steps["model"], "coef_"):
        coef = np.asarray(baseline_pipe.named_steps["model"].coef_)
        order = np.argsort(np.abs(coef))[-topk_by_coef:][::-1]
        cols = [X_cols[i] for i in order]
        drops.append((f"topk_coef:{topk_by_coef}", cols))

    return drops


def run_label_permutations(args: argparse.Namespace, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(args.random_state)
    rows: List[Dict[str, float]] = []
    for i in range(args.n_label_perm):
        perm_idx = rng.permutation(len(y_train))
        y_perm = y_train.values[perm_idx]
        pipe = build_model(args)
        m, _ = fit_eval(pipe, X_train, pd.Series(y_perm), X_test, y_test)
        rows.append({"iter": i, **m})
    return pd.DataFrame(rows)


def run_feature_shuffles(args: argparse.Namespace, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(args.random_state + 1337)
    rows: List[Dict[str, float]] = []
    for i in range(args.n_feat_shuffle):
        X_tr = X_train.copy()
        for c in X_tr.columns:
            X_tr[c] = X_tr[c].values[rng.permutation(len(X_tr))]
        pipe = build_model(args)
        m, _ = fit_eval(pipe, X_tr, y_train, X_test, y_test)
        rows.append({"iter": i, **m})
    return pd.DataFrame(rows)


def run_ablations(args: argparse.Namespace, base_pipe: Pipeline,
                  X_train: pd.DataFrame, y_train: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series,
                  baseline_metrics: Dict[str, float]) -> pd.DataFrame:
    X_cols = list(X_train.columns)
    dropsets = collect_features_to_drop(
        X_cols=X_cols,
        regex_list=args.ablate_regex,
        list_file=args.ablate_list,
        topk_by_coef=args.ablate_topk_coef,
        baseline_pipe=base_pipe,
    )
    rows: List[Dict[str, float]] = []
    for name, cols in dropsets:
        keep = [c for c in X_cols if c not in cols]
        Xtr2, Xte2 = X_train[keep], X_test[keep]
        pipe = build_model(args)
        m, _ = fit_eval(pipe, Xtr2, y_train, Xte2, y_test)
        rows.append({
            "ablation": name,
            "removed_n": len(cols),
            "kept_n": len(keep),
            **m,
            "delta_r2": m["r2"] - baseline_metrics["r2"],
            "delta_rmse": m["rmse"] - baseline_metrics["rmse"],
            "delta_mae": m["mae"] - baseline_metrics["mae"],
            "delta_spearman": m["spearman"] - baseline_metrics["spearman"],
        })
    return pd.DataFrame(rows)


# ------------------------ Main ------------------------

def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Utility: list columns and exit
    if args.list_columns:
        for which, p in [("train", args.train_csv), ("test", args.test_csv)]:
            try:
                cols = list(pd.read_csv(p, nrows=0).columns)
            except Exception as e:
                cols = [f"<failed to read: {e}>"]
            print(f"{which} columns ({p}):")
            print(cols)
        sys.exit(0)

    # Load data
    X_train, y_train, train_ids = load_matrix(args.train_csv, args.target_col, args.id_col)
    X_test, y_test, test_ids = load_matrix(args.test_csv, args.target_col, args.id_col)

    # Baseline
    base_pipe = build_model(args)
    t0 = time.time()
    baseline_metrics, y_pred = fit_eval(base_pipe, X_train, y_train, X_test, y_test)
    dur = time.time() - t0

    print("Baseline:", baseline_metrics)
    write_json({"metrics": baseline_metrics, "seconds": dur}, outdir / "baseline_metrics.json")
    if args.save_preds:
        dfp = pd.DataFrame({
            "y_true": y_test.values,
            "y_pred": y_pred,
        })
        if test_ids is not None:
            dfp.insert(0, "id", test_ids.values)
        write_csv(dfp, outdir / "baseline_predictions.csv")

    # Negative controls
    label_df = run_label_permutations(args, X_train, y_train, X_test, y_test)
    write_csv(label_df, outdir / "label_permutations.csv")
    plot_hist(label_df["r2"].values, baseline_metrics["r2"], xlabel="R^2",
              title="Label permutation: R^2 distribution",
              outpath=outdir / "label_perm_r2_hist.png")

    shuffle_df = run_feature_shuffles(args, X_train, y_train, X_test, y_test)
    write_csv(shuffle_df, outdir / "feature_shuffles.csv")
    plot_hist(shuffle_df["r2"].values, baseline_metrics["r2"], xlabel="R^2",
              title="Feature column shuffle (train only): R^2 distribution",
              outpath=outdir / "feature_shuffle_r2_hist.png")

    # Ablations
    ablate_df = run_ablations(args, base_pipe, X_train, y_train, X_test, y_test, baseline_metrics)
    if not ablate_df.empty:
        write_csv(ablate_df, outdir / "ablations.csv")
        import matplotlib.pyplot as plt
        sdf = ablate_df.sort_values("delta_r2")  # negative values mean degradation
        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(sdf))))
        ax.barh(sdf["ablation"], sdf["delta_r2"])
        ax.axvline(0.0, color="grey", lw=1)
        ax.set_xlabel("Δ R^2 vs baseline (negative = worse)")
        fig.tight_layout()
        fig.savefig(outdir / "ablations_delta_r2.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Summary
    def pctl(a: np.ndarray, q: float) -> float:
        return float(np.percentile(a, q)) if len(a) else float("nan")

    summary = {
        "baseline": baseline_metrics,
        "label_perm": {
            "n": int(len(label_df)),
            "r2_mean": float(label_df["r2"].mean()) if len(label_df) else float("nan"),
            "r2_p95": pctl(label_df["r2"].values, 95),
        },
        "feature_shuffle": {
            "n": int(len(shuffle_df)),
            "r2_mean": float(shuffle_df["r2"].mean()) if len(shuffle_df) else float("nan"),
            "r2_p95": pctl(shuffle_df["r2"].values, 95),
        },
        "ablations": None if ablate_df.empty else ablate_df.to_dict(orient="records"),
    }
    write_json(summary, outdir / "summary.json")

    print("Done. Outputs in:", outdir)


if __name__ == "__main__":
    main()
