#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Wrapper: run_test_pipeline.sh
# Defaults for test_samples, but overrideable via CLI.
# -----------------------------------------------------------------------------

# Default values
SAMPLES="test_samples/*.fasta"
REFERENCE="raw_data/NC_045512.2_sequence.fasta"
TRAIN_MATRIX="lasso_training_data/feature_matrix_train.csv"
MODEL="model_artifacts/lasso_model.joblib"
SCALER="model_artifacts/scaler.joblib"
OUTDIR="results_test"
CHUNK_SIZE=10
IDENTITY_THRESH=92.0
PROFILE="standard"

usage() {
  cat <<EOF
Usage: $0 [options]

  --samples PATTERN            glob or path for FASTA files (default: $SAMPLES)
  --reference_fasta FILE       reference FASTA (default: $REFERENCE)
  --train_feature_matrix FILE  training feature-matrix CSV (default: $TRAIN_MATRIX)
  --model FILE                 trained model .joblib (default: $MODEL)
  --scaler FILE                scaler .joblib (default: $SCALER)
  --outdir DIR                 output directory (default: $OUTDIR)
  --chunk_size INT             chunk size (default: $CHUNK_SIZE)
  --identity_thresh FLOAT      pct identity threshold (default: $IDENTITY_THRESH)
  --profile NAME               nextflow profile (default: $PROFILE)
  -h, --help                   show this help and exit

Examples:
  # run with defaults:
  $0

  # override only samples and outdir:
  $0 --samples \"my_samples/*.fa\" --outdir my_results
EOF
  exit 1
}

# Parse CLI arguments
while (( $# )); do
  case "$1" in
    --samples)            SAMPLES="$2"; shift 2 ;;
    --reference_fasta)    REFERENCE="$2"; shift 2 ;;
    --train_feature_matrix) TRAIN_MATRIX="$2"; shift 2 ;;
    --model)              MODEL="$2"; shift 2 ;;
    --scaler)             SCALER="$2"; shift 2 ;;
    --outdir)             OUTDIR="$2"; shift 2 ;;
    --chunk_size)         CHUNK_SIZE="$2"; shift 2 ;;
    --identity_thresh)    IDENTITY_THRESH="$2"; shift 2 ;;
    --profile)            PROFILE="$2"; shift 2 ;;
    -h|--help)            usage ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Show run settings
echo "[INFO] Running pipeline with settings:"
echo "       samples            = $SAMPLES"
echo "       reference_fasta    = $REFERENCE"
echo "       train_feature_matrix = $TRAIN_MATRIX"
echo "       model              = $MODEL"
echo "       scaler             = $SCALER"
echo "       outdir             = $OUTDIR"
echo "       chunk_size         = $CHUNK_SIZE"
echo "       identity_thresh    = $IDENTITY_THRESH"
echo "       profile            = $PROFILE"
echo

# Launch Nextflow
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
