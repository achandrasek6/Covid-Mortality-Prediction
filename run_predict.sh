#!/usr/bin/env bash
set -euo pipefail

# ---- config defaults ----
VENV="myenv"
DEFAULT_SAMPLES="transformed_data/variant_samples_small.fasta"
DEFAULT_REFERENCE="raw_data/NC_045512.2_sequence.fasta"
DEFAULT_TRAIN_MATRIX="lasso_training_data/feature_matrix_train.csv"
DEFAULT_MODEL="model_artifacts/lasso_model.joblib"
DEFAULT_SCALER="model_artifacts/scaler.joblib"
DEFAULT_OUTDIR="analysis_output"

# ---- CLI parsing ----
usage() {
  cat <<EOF
Usage: $0 [options]

Wrapper to run the Nextflow prediction pipeline.

Options:
  --samples PATH           Sample FASTA (default: ${DEFAULT_SAMPLES})
  --reference PATH         Reference FASTA (default: ${DEFAULT_REFERENCE})
  --train-matrix PATH      Training feature matrix (default: ${DEFAULT_TRAIN_MATRIX})
  --model PATH             Saved lasso model (default: ${DEFAULT_MODEL})
  --scaler PATH            Saved scaler (default: ${DEFAULT_SCALER})
  --outdir PATH            Output directory (default: ${DEFAULT_OUTDIR})
  --no-activate            Skip virtualenv activation
  -h, --help               Show this help message
EOF
  exit 1
}

# defaults
samples=${DEFAULT_SAMPLES}
reference=${DEFAULT_REFERENCE}
train_matrix=${DEFAULT_TRAIN_MATRIX}
model=${DEFAULT_MODEL}
scaler=${DEFAULT_SCALER}
outdir=${DEFAULT_OUTDIR}
activate_venv=1

while [[ $# -gt 0 ]]; do
  case $1 in
    --samples) samples="$2"; shift 2 ;;
    --reference) reference="$2"; shift 2 ;;
    --train-matrix) train_matrix="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --scaler) scaler="$2"; shift 2 ;;
    --outdir) outdir="$2"; shift 2 ;;
    --no-activate) activate_venv=0; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

# Activate virtualenv unless skipped
if [[ $activate_venv -eq 1 ]]; then
  if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${VENV}/bin/activate"
  else
    echo "Virtualenv not found at ${VENV}/bin/activate" >&2
    exit 1
  fi
fi

echo "Running prediction pipeline with:"
echo "  samples:       $samples"
echo "  reference:     $reference"
echo "  train matrix:  $train_matrix"
echo "  model:         $model"
echo "  scaler:        $scaler"
echo "  outdir:        $outdir"

nextflow run main.nf \
  --samples "$samples" \
  --reference_fasta "$reference" \
  --train_feature_matrix "$train_matrix" \
  --model "$model" \
  --scaler "$scaler" \
  --outdir "$outdir"
