#!/usr/bin/env bash

# run_pipeline.sh - wrapper for Nextflow COVID-19 CFR prediction workflow
#
# By default, uses the built-in paths; you can override any via flags.
#
# Defaults:
#   SAMPLES="transformed_data/variant_samples_small.fasta"
#   REFERENCE_FASTA="raw_data/NC_045512.2_sequence.fasta"
#   TRAIN_FEATURE_MATRIX="lasso_training_data/feature_matrix_train.csv"
#   MODEL="model_artifacts/lasso_model.joblib"
#   SCALER="model_artifacts/scaler.joblib"
#   CHUNK_SIZE=10
#   IDENTITY_THRESH=92.0
#   OUTDIR="analysis_output"
#   PROFILE="standard"
#
# Usage:
#   ./run_pipeline.sh [options]
#
# Options:
#   --samples <FASTA>               Override default samples FASTA
#   --reference-fasta <FASTA>       Override default reference FASTA
#   --train-feature-matrix <CSV>    Override default train-feature matrix
#   --model <JOBLIB>                Override default Lasso model
#   --scaler <JOBLIB>               Override default scaler
#   --chunk-size <int>              Override default chunk size
#   --identity-thresh <float>       Override default identity threshold
#   --outdir <dir>                  Override default output directory
#   --profile <name>                Override default Nextflow profile
#   -h, --help                      Show this help message and exit

set -eo pipefail

# Default values
SAMPLES="transformed_data/variant_samples_small.fasta"
REFERENCE_FASTA="raw_data/NC_045512.2_sequence.fasta"
TRAIN_FEATURE_MATRIX="lasso_training_data/feature_matrix_train.csv"
MODEL="model_artifacts/lasso_model.joblib"
SCALER="model_artifacts/scaler.joblib"
CHUNK_SIZE=10
IDENTITY_THRESH=92.0
OUTDIR=""
PROFILE="standard"

usage() {
  sed -n '1,20p' "$0"
  exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --samples)               SAMPLES="$2"; shift 2;;
    --reference-fasta)       REFERENCE_FASTA="$2"; shift 2;;
    --train-feature-matrix)  TRAIN_FEATURE_MATRIX="$2"; shift 2;;
    --model)                 MODEL="$2"; shift 2;;
    --scaler)                SCALER="$2"; shift 2;;
    --chunk-size)            CHUNK_SIZE="$2"; shift 2;;
    --identity-thresh)       IDENTITY_THRESH="$2"; shift 2;;
    --outdir)                OUTDIR="$2"; shift 2;;
    --profile)               PROFILE="$2"; shift 2;;
    -h|--help)               usage;;
    *) echo "Unknown option: $1" >&2; usage;;
  esac
done

# If OUTDIR wasn’t given (or was left empty), derive it from the FASTA basename:
if [ -z "$OUTDIR" ]; then
  # strip any path, then strip extension
  sample_base=$(basename "$SAMPLES")
  sample_base="${sample_base%%.*}"
  OUTDIR="${sample_base}_out"
fi

# Execute Nextflow workflow with chosen or default parameters
exec nextflow run main.nf \
  --samples               "$SAMPLES" \
  --reference_fasta       "$REFERENCE_FASTA" \
  --train_feature_matrix  "$TRAIN_FEATURE_MATRIX" \
  --model                 "$MODEL" \
  --scaler                "$SCALER" \
  --chunk_size            "$CHUNK_SIZE" \
  --identity_thresh       "$IDENTITY_THRESH" \
  --outdir                "$OUTDIR" \
  -profile                "$PROFILE"
