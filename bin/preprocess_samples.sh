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
OUTDIR="$PROJECT_ROOT/preprocessed"
IDENTITY=92.0
ACTIVATE=1

# ---- usage ----
usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Wrapper to run only the preprocessing step.

Options:
  --samples PATH          Sample FASTA (default: $SAMPLES)
  --reference PATH        Reference FASTA (default: $REFERENCE)
  --identity-threshold N  Identity threshold (default: $IDENTITY)
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
    --identity-threshold) IDENTITY="$2"; shift 2 ;;
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

# ---- run preprocessing ----
echo "Project root: $PROJECT_ROOT"
echo "Running preprocessing with:"
echo "  samples:             $SAMPLES"
echo "  reference:           $REFERENCE"
echo "  identity-threshold:  $IDENTITY"
echo "  outdir:              $OUTDIR"

python3 "$PROJECT_ROOT/scripts/preprocess_all.py" \
  --samples "$SAMPLES" \
  --reference-fasta "$REFERENCE" \
  --identity-threshold "$IDENTITY" \
  --out-dir "$OUTDIR"
