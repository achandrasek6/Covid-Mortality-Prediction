#!/usr/bin/env python3
"""
ML_model.py

Train a Lasso regression model on precomputed feature matrices and save artifacts.

Responsibilities:
  1. Load training/testing matrices including identifiers, variant features, and target (Global CFR).
  2. Scale features with StandardScaler (configurable 'with_mean').
  3. Train Lasso with user-specified alpha.
  4. Save model, scaler, and feature list to an output directory.
  5. Evaluate performance (R² and MSE) on train and test sets.
  6. Print retained features and coefficients.

Typical usage:
  python3 ML_model_user_CLI.py \
    --train-matrix lasso_training_data/feature_matrix_train.csv \
    --test-matrix  lasso_training_data/feature_matrix_test.csv  \
    --alpha 0.000174 \
    --out-dir model_artifacts \
    [--with-mean]
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib

def load_data(path):
    df = pd.read_csv(path)
    feature_cols = df.columns.drop(["SampleID", "Variant", "Global CFR"])
    X = df[feature_cols].values
    y = df["Global CFR"].values
    return X, y, feature_cols

def train_and_scale(X_train, y_train, alpha, with_mean=False):
    scaler = StandardScaler(with_mean=with_mean)
    X_train_s = scaler.fit_transform(X_train)
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train_s, y_train)
    return model, scaler

def evaluate(model, scaler, X, y):
    X_s = scaler.transform(X)
    y_pred = model.predict(X_s)
    return r2_score(y, y_pred), mean_squared_error(y, y_pred), y_pred

def save_artifacts(model, scaler, feature_cols, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "lasso_model.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    joblib.dump(feature_cols.tolist(), os.path.join(out_dir, "feature_cols.joblib"))

def print_results(train_metrics, test_metrics, feature_cols, model):
    train_r2, train_mse, _ = train_metrics
    test_r2, test_mse, _  = test_metrics
    print(f"Training R²: {train_r2:.3f}")
    print(f"Training MSE: {train_mse:.3e}")
    print(f"Testing  R²: {test_r2:.3f}")
    print(f"Testing  MSE: {test_mse:.3e}")

    coefs = model.coef_
    selected_idx = np.where(coefs != 0)[0]
    selected_features = np.array(feature_cols)[selected_idx]
    selected_coefs    = coefs[selected_idx]

    print(f"\nNumber of features retained: {len(selected_features)}")
    print("Selected features and coefficients:")
    for feat, coef in zip(selected_features, selected_coefs):
        print(f"  {feat:15s} → {coef:.6f}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Lasso regression model on feature matrices."
    )
    parser.add_argument(
        "--train-matrix", required=True,
        help="Path to the training matrix CSV."
    )
    parser.add_argument(
        "--test-matrix", required=True,
        help="Path to the testing matrix CSV."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.000174,
        help="Regularization strength for Lasso."
    )
    parser.add_argument(
        "--out-dir", default="model_artifacts",
        help="Directory to save model artifacts."
    )
    parser.add_argument(
        "--with-mean", action="store_true",
        help="Enable StandardScaler with_mean=True (default False)."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    X_train, y_train, feature_cols = load_data(args.train_matrix)
    X_test,  y_test, _            = load_data(args.test_matrix)

    model, scaler = train_and_scale(
        X_train, y_train, alpha=args.alpha,
        with_mean=args.with_mean
    )

    train_metrics = evaluate(model, scaler, X_train, y_train)
    test_metrics  = evaluate(model, scaler, X_test,  y_test)

    save_artifacts(model, scaler, feature_cols, args.out_dir)
    print_results(train_metrics, test_metrics, feature_cols, model)

if __name__ == "__main__":
    main()
