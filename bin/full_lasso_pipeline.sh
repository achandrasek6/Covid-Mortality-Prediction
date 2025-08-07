#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Wrapper for Nextflow lasso pipeline with sensible defaults for test samples
# -----------------------------------------------------------------------------

# Default parameters
SAMPLES="test_samples/*.fasta"
REFERENCE="raw_data/NC_045512.2_sequence.fasta"
TRAIN_MATRIX="lasso_training_data/feature_matrix_train.csv"
MODEL="model_artifacts/lasso_model.joblib"
SCALER="model_artifacts/scaler.joblib"
OUTDIR="results_test"
CHUNK_SIZE=10
IDENTITY_THRESH=92.0
PROFILE="standard"

# You can override any of the above by exporting e.g.
#   export SAMPLES="other_folder/*.fasta"
# or by editing this script directly.

echo "[INFO] Running pipeline with:"
echo "       samples           = $SAMPLES"
echo "       reference_fasta   = $REFERENCE"
echo "       train_feature_csv = $TRAIN_MATRIX"
echo "       model             = $MODEL"
echo "       scaler            = $SCALER"
echo "       outdir            = $OUTDIR"
echo "       chunk_size        = $CHUNK_SIZE"
echo "       identity_thresh   = $IDENTITY_THRESH"
echo "       profile           = $PROFILE"

nextflow run main.nf \
  --samples           "$SAMPLES" \
  --reference_fasta   "$REFERENCE" \
  --train_feature_matrix "$TRAIN_MATRIX" \
  --model             "$MODEL" \
  --scaler            "$SCALER" \
  --outdir            "$OUTDIR" \
  --chunk_size        "$CHUNK_SIZE" \
  --identity_thresh   "$IDENTITY_THRESH" \
  -profile "$PROFILE"
