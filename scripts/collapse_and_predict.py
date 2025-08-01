#!/usr/bin/env python3
"""
collapse_and_predict.py

Pipeline step that takes preprocessed sequence data, collapses it into the feature
space expected by the trained Lasso model, loads the saved model and scaler, and
produces mortality-rate predictions (e.g., CFR). Optionally, if ground-truth labels
are provided, it computes evaluation metrics.

Main responsibilities:
  1. Load the filtered/ aligned variant matrix and aligned FASTA.
  2. Collapse the variant representation to the same feature set as used in training,
     respecting the selected features (e.g., via the original training feature matrix).
  3. Load the persisted scaler and Lasso model (from model_artifacts).
  4. Apply scaling to the new data and run model.predict to get CFR/mortality predictions.
  5. If a target CSV is supplied, compare predictions to ground truth and output metrics.
  6. Write out:
       - Collapsed feature matrix (matching model input)
       - Predictions CSV
       - Metrics file (if applicable)

Expected inputs (via CLI arguments):
  --variant-matrix           : Binary variant matrix produced by preprocessing.
  --aligned-fasta           : Filtered & aligned FASTA used to derive sample ordering.
  --reference-id            : Reference sequence ID (used for alignment sanity).
  --train-feature-matrix    : Original training feature matrix (used to determine selected features and column headers).
  --model                   : Path to saved Lasso model (e.g., lasso_model.joblib).
  --scaler                  : Path to saved scaler (e.g., scaler.joblib).
  --target                 : Optional CSV with true labels (e.g., SampleID and Global CFR) for computing metrics.
  --out-dir                : Output directory to write predictions, collapsed matrix, and metrics.

Outputs:
  final_predictions/predictions.csv               : Model output per sample.
  final_predictions/collapsed_feature_matrix.csv  : Feature matrix used for prediction (subset of training features).
  final_predictions/metrics.txt (if --target)     : Evaluation metrics (e.g., R², MSE) comparing predictions to truth.

Example usage:
  python3 scripts/collapse_and_predict.py \
    --variant-matrix preprocessed/variant_binary_matrix.csv \
    --aligned-fasta preprocessed/aligned_filtered.fasta \
    --reference-id NC_045512.2 \
    --train-feature-matrix lasso_training_data/feature_matrix_train.csv \
    --model model_artifacts/lasso_model.joblib \
    --scaler model_artifacts/scaler.joblib \
    --target data/true_cfrs.csv \
    --out-dir final_predictions
"""
import argparse
import os
import re
import logging
import pandas as pd
import joblib
import numpy as np
from Bio import AlignIO
from sklearn.metrics import mean_squared_error, r2_score

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def build_gene_position_mapping(aligned_fasta, ref_id):
    alignment = AlignIO.read(aligned_fasta, "fasta")
    ref_record = next((r for r in alignment if ref_id in r.id), None)
    if ref_record is None:
        raise RuntimeError(f"Reference '{ref_id}' not found in {aligned_fasta}")

    # Gene annotation intervals (reference coordinates, 1-based)
    gene_annotations = [
        {"gene": "ORF1ab", "start": 265, "end": 21555},
        {"gene": "S",      "start": 21562, "end": 25384},
        {"gene": "ORF3a",  "start": 25392, "end": 26220},
        {"gene": "E",      "start": 26244, "end": 26472},
        {"gene": "M",      "start": 26522, "end": 27191},
        {"gene": "ORF6",   "start": 27201, "end": 27387},
        {"gene": "ORF7a",  "start": 27393, "end": 27759},
        {"gene": "ORF7b",  "start": 27755, "end": 27887},
        {"gene": "ORF8",   "start": 27893, "end": 28259},
        {"gene": "N",      "start": 28273, "end": 29533},
        {"gene": "ORF10",  "start": 29557, "end": 29674}
    ]

    # Map alignment index -> reference genomic position (skipping gaps)
    mapping = {}
    ref_pos = 0
    for idx, nt in enumerate(ref_record.seq):
        if nt != "-":
            ref_pos += 1
            mapping[idx] = ref_pos
        else:
            mapping[idx] = None

    # For each gene, get alignment indices that fall in its reference coordinate interval
    gene_alignment_indices = {}
    for ga in gene_annotations:
        name = ga["gene"]
        indices = [i for i, rp in mapping.items() if rp is not None and ga["start"] <= rp <= ga["end"]]
        gene_alignment_indices[name] = indices

    # Reverse map: alignment index -> gene_relative label like "S_12"
    alignidx_to_genepos = {}
    for gene, indices in gene_alignment_indices.items():
        for local_pos, aln_idx in enumerate(indices, start=1):
            label = f"{gene}_{local_pos}"
            alignidx_to_genepos[aln_idx] = label

    return alignidx_to_genepos

def collapse_variants(variant_df, alignidx_to_genepos, selected_features):
    # Initialize collapsed DataFrame with selected features (0)
    collapsed = pd.DataFrame(0, index=variant_df.index, columns=selected_features, dtype=int)

    for col in variant_df.columns:
        m = re.match(r"pos(\d+)_", col)
        if not m:
            continue
        aln_pos = int(m.group(1)) - 1  # pos123 -> alignment index 122
        gene_feat = alignidx_to_genepos.get(aln_pos)
        if gene_feat and gene_feat in selected_features:
            # wherever this detailed variant is 1, mark the gene-level feature as 1
            mask = variant_df[col] == 1
            collapsed.loc[mask, gene_feat] = 1

    return collapsed

def align_and_predict(args):
    setup_logger()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load training feature names to know which features the model expects (selected 64)
    logging.info("Loading original training feature matrix to extract feature names.")
    train_df = pd.read_csv(args.train_feature_matrix)
    train_features = [c for c in train_df.columns if c not in ["SampleID", "Variant", "Global CFR"]]

    # Load new high-resolution variant matrix
    logging.info("Loading new variant binary matrix.")
    variant_df = pd.read_csv(args.variant_matrix, index_col=0)

    # Build gene-relative mapping from alignment
    logging.info("Building gene-relative position mapping from aligned FASTA.")
    alignidx_to_genepos = build_gene_position_mapping(args.aligned_fasta, args.reference_id)

    # Collapse high-res variants into gene-pos features
    logging.info("Collapsing detailed variants into gene-relative features.")
    collapsed_df = collapse_variants(variant_df, alignidx_to_genepos, train_features)

    # Load scaler and model
    logging.info("Loading scaler and model.")
    scaler = joblib.load(args.scaler)
    model = joblib.load(args.model)

    # Scale and predict
    X = collapsed_df.values
    X_s = scaler.transform(X)
    y_pred = model.predict(X_s)

    # Build output DataFrame
    out_df = pd.DataFrame({
        "SampleID": collapsed_df.index,
        "predicted_Global_CFR": y_pred
    })

    # If true targets provided, merge and compute metrics
    if args.target:
        logging.info("Loading true target values for evaluation.")
        target_df = pd.read_csv(args.target).set_index("SampleID")
        if "Global CFR" not in target_df.columns:
            raise ValueError("Target CSV must contain column 'Global CFR'")
        y_true = target_df.loc[collapsed_df.index, "Global CFR"].values
        out_df["true_Global_CFR"] = y_true
        out_df["residual"] = y_true - y_pred
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        with open(os.path.join(args.out_dir, "metrics.txt"), "w") as f:
            f.write(f"mse: {mse}\n")
            f.write(f"r2: {r2}\n")
        logging.info(f"Prediction metrics -- MSE: {mse:.3e}, R²: {r2:.3f}")

    # Save collapsed feature matrix and predictions
    collapsed_path = os.path.join(args.out_dir, "collapsed_feature_matrix.csv")
    pred_path = os.path.join(args.out_dir, "predictions.csv")
    collapsed_df.to_csv(collapsed_path)
    out_df.to_csv(pred_path, index=False)
    logging.info(f"Saved collapsed features to {collapsed_path}")
    logging.info(f"Saved predictions to {pred_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Collapse new variant matrix to original gene-pos features and predict with existing Lasso model.")
    parser.add_argument("--variant-matrix", required=True, help="High-res binary variant matrix (e.g., new_prep/variant_binary_matrix.csv).")
    parser.add_argument("--aligned-fasta", required=True, help="Aligned FASTA (must contain reference and be same alignment used for gene-pos mapping).")
    parser.add_argument("--reference-id", default="NC_045512.2", help="Reference sequence ID in the aligned FASTA.")
    parser.add_argument("--train-feature-matrix", required=True, help="Original training feature matrix (to get the 64 selected feature names).")
    parser.add_argument("--model", required=True, help="Path to saved Lasso model (.joblib).")
    parser.add_argument("--scaler", required=True, help="Path to saved scaler (.joblib).")
    parser.add_argument("--target", help="Optional CSV with true 'Global CFR' for evaluation (columns: SampleID, Global CFR).")
    parser.add_argument("--out-dir", default="collapsed_prediction", help="Output directory.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    align_and_predict(args)
