#!/usr/bin/env bash
set -euo pipefail

# ---- helper: find project root (looks for nextflow.config or .git) ----
find_root() {
  local dir
  dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # assumes script is in bin/
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/nextflow.config" || -d "$dir/.git" ]]; then
      echo "$dir"
      return
    fi
    dir=$(dirname "$dir")
  done
  echo "$(pwd)"  # fallback
}

# ---- determine project root with explicit fallback warning ----
CANDIDATE_ROOT=$(find_root)
if [[ -f "$CANDIDATE_ROOT/nextflow.config" || -d "$CANDIDATE_ROOT/.git" ]]; then
  PROJECT_ROOT="$CANDIDATE_ROOT"
else
  echo "Warning: project root heuristic failed; using fallback directory: $CANDIDATE_ROOT" >&2
  PROJECT_ROOT="$CANDIDATE_ROOT"
fi

VENV="$PROJECT_ROOT/myenv"

# ---- defaults ----
SAMPLES="$PROJECT_ROOT/transformed_data/variant_samples_small.fasta"
REFERENCE="$PROJECT_ROOT/raw_data/NC_045512.2_sequence.fasta"
TRAIN_MATRIX="$PROJECT_ROOT/lasso_training_data/feature_matrix_train.csv"
MODEL="$PROJECT_ROOT/model_artifacts/lasso_model.joblib"
SCALER="$PROJECT_ROOT/model_artifacts/scaler.joblib"
OUTDIR="$PROJECT_ROOT/analysis_output"
ACTIVATE=1

# ---- usage ----
usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Wrapper to run the Nextflow prediction pipeline.

Options:
  --samples PATH          Sample FASTA (default: $SAMPLES)
  --reference PATH        Reference FASTA (default: $REFERENCE)
  --train-matrix PATH     Training feature matrix (default: $TRAIN_MATRIX)
  --model PATH            Lasso model artifact (default: $MODEL)
  --scaler PATH           Scaler artifact (default: $SCALER)
  --outdir PATH           Output directory (default: $OUTDIR)
  --no-activate           Skip virtualenv activation
  -h, --help              Show this help message
EOF
  exit 1
}

# ---- parse args ----
while [[ $# -gt 0 ]]; do
  case $1 in
    --samples) SAMPLES="$2"; shift 2 ;;
    --reference) REFERENCE="$2"; shift 2 ;;
    --train-matrix) TRAIN_MATRIX="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --scaler) SCALER="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --no-activate) ACTIVATE=0; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

# ---- activate virtualenv if requested ----
if [[ $ACTIVATE -eq 1 ]]; then
  if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$VENV/bin/activate"
  else
    echo "Warning: virtualenv not found at $VENV/bin/activate; continuing without activation" >&2
  fi
fi

# ---- run pipeline ----
echo "Project root: $PROJECT_ROOT"
echo "Running prediction pipeline with:"
echo "  samples:       $SAMPLES"
echo "  reference:     $REFERENCE"
echo "  train matrix:  $TRAIN_MATRIX"
echo "  model:         $MODEL"
echo "  scaler:        $SCALER"
echo "  outdir:        $OUTDIR"

nextflow run "$PROJECT_ROOT/main.nf" \
  --samples "$SAMPLES" \
  --reference_fasta "$REFERENCE" \
  --train_feature_matrix "$TRAIN_MATRIX" \
  --model "$MODEL" \
  --scaler "$SCALER" \
  --outdir "$OUTDIR"
