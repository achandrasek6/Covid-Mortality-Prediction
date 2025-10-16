#!/usr/bin/env bash
set -euo pipefail

echo "== Nextflow runner starting =="
echo "NXF_HOME: ${NXF_HOME:-unset}"
echo "AWS_REGION: ${AWS_REGION:-unset}"
echo "PROFILE: ${PROFILE:-aws}"
echo "Container tag: ${CONTAINER_TAG:-unset}"
echo "Container digest: ${CONTAINER_DIGEST:-unset}"

# Required-ish params can be provided via env or CLI args:
#   SAMPLES, OUTDIR, REFERENCE_FASTA, TRAIN_FEATURE_MATRIX, MODEL, SCALER
# You can still override everything by passing raw args to the container.

NF_ARGS=()

# Choose profile (default aws)
PROFILE="${PROFILE:-aws}"
NF_ARGS+=("-profile" "$PROFILE")

# Allow passing container tag/digest to the pipeline params
if [[ -n "${CONTAINER_TAG:-}" ]]; then
  NF_ARGS+=("--container_tag" "${CONTAINER_TAG}")
fi
if [[ -n "${CONTAINER_DIGEST:-}" ]]; then
  # pass just the hex without "sha256:"; config already adds it
  NF_ARGS+=("--container_digest" "${CONTAINER_DIGEST}")
fi

# Wire common S3 params if present
[[ -n "${SAMPLES:-}" ]]              && NF_ARGS+=("--samples" "${SAMPLES}")
[[ -n "${OUTDIR:-}" ]]               && NF_ARGS+=("--outdir" "${OUTDIR}")
[[ -n "${REFERENCE_FASTA:-}" ]]      && NF_ARGS+=("--reference_fasta" "${REFERENCE_FASTA}")
[[ -n "${TRAIN_FEATURE_MATRIX:-}" ]] && NF_ARGS+=("--train_feature_matrix" "${TRAIN_FEATURE_MATRIX}")
[[ -n "${MODEL:-}" ]]                && NF_ARGS+=("--model" "${MODEL}")
[[ -n "${SCALER:-}" ]]               && NF_ARGS+=("--scaler" "${SCALER}")

# Verbose toggle
if [[ "${VERBOSE:-false}" == "true" ]]; then
  NF_ARGS+=("--verbose")
fi

# If the user supplied explicit args to the container, append them last to allow overrides
if [[ "$#" -gt 0 ]]; then
  NF_ARGS+=("$@")
fi

echo "Running: nextflow run /pipeline/main.nf ${NF_ARGS[*]}"
exec /home/appuser/bin/nextflow run /pipeline/main.nf "${NF_ARGS[@]}"
