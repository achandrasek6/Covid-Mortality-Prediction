#!/usr/bin/env bash
set -euo pipefail

# ---- helper: find project root (looks for nextflow.config or .git) ----
find_root() {
  local dir
  dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # assumes script lives in bin/
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
VARIANT_MATRIX="$PROJECT_ROOT/preprocessed/variant_binary_matrix.csv"
ALIGNED_FASTA="$PROJECT_ROOT/preprocessed/aligned_filtered.fasta"
REFERENCE_ID="NC_045512.2"
TRAIN_MATRIX="$PROJECT_ROOT/lasso_training_data/feature_matrix_train.csv"
MODEL="$PROJECT_ROOT/model_artifacts/lasso_model.joblib"
SCALER="$PROJECT_ROOT/model_artifacts/scaler.joblib"
TARGET=""  # optional
OUTDIR="$PROJECT_ROOT/final_predictions"
ACTIVATE=1

# ---- usage ----
usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Wrapper to run collapse_and_predict step.

Options:
  --variant-matrix PATH       Binary variant matrix (default: $VARIANT_MATRIX)
  --aligned-fasta PATH        Filtered & aligned FASTA (default: $ALIGNED_FASTA)
  --reference-id ID           Reference sequence ID (default: $REFERENCE_ID)
  --train-matrix PATH         Training feature matrix (default: $TRAIN_MATRIX)
  --model PATH                Lasso model artifact (default: $MODEL)
  --scaler PATH               Scaler artifact (default: $SCALER)
  --target PATH              Optional ground truth CSV for metrics
  --outdir PATH              Output directory (default: $OUTDIR)
  --no-activate              Skip virtualenv activation
  -h, --help                 Show this help message
EOF
  exit 1
}

# ---- parse args ----
while [[ $# -gt 0 ]]; do
  case $1 in
    --variant-matrix) VARIANT_MATRIX="$2"; shift 2 ;;
    --aligned-fasta) ALIGNED_FASTA="$2"; shift 2 ;;
    --reference-id) REFERENCE_ID="$2"; shift 2 ;;
    --train-matrix) TRAIN_MATRIX="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --scaler) SCALER="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
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

# ---- run collapse & predict ----
echo "Project root: $PROJECT_ROOT"
echo "Running collapse_and_predict with:"
echo "  variant matrix:        $VARIANT_MATRIX"
echo "  aligned fasta:         $ALIGNED_FASTA"
echo "  reference ID:          $REFERENCE_ID"
echo "  train matrix:          $TRAIN_MATRIX"
echo "  model:                 $MODEL"
echo "  scaler:                $SCALER"
if [[ -n "$TARGET" ]]; then
  echo "  target:                $TARGET"
fi
echo "  outdir:                $OUTDIR"

mkdir -p "$OUTDIR"

target_arg=""
if [[ -n "$TARGET" ]]; then
  target_arg="--target $TARGET"
fi

python3 "$PROJECT_ROOT/scripts/collapse_and_predict.py" \
  --variant-matrix "$VARIANT_MATRIX" \
  --aligned-fasta "$ALIGNED_FASTA" \
  --reference-id "$REFERENCE_ID" \
  --train-feature-matrix "$TRAIN_MATRIX" \
  --model "$MODEL" \
  --scaler "$SCALER" \
  $target_arg \
  --out-dir "$OUTDIR"
