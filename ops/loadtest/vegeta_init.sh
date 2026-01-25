#!/usr/bin/env bash
set -euo pipefail

: "${API_BASE:?set API_BASE (e.g. https://...execute-api.../dev)}"
: "${API_KEY:?set API_KEY (API key VALUE used for x-api-key)}"

OUTDIR="${OUTDIR:-s3://ach-covid-lasso-us-east-2/results}"
RATE="${RATE:-5}"          # req/sec
DURATION="${DURATION:-30s}"

BODY_FILE=/tmp/vegeta_body.json
cat > "$BODY_FILE" <<EOF
{
  "phase": "init",
  "reference_fasta": "s3://ach-covid-lasso-us-east-2/inputs/reference/NC_045512.2_sequence.fasta",
  "train_feature_matrix": "s3://ach-covid-lasso-us-east-2/inputs/lasso/feature_matrix_train.csv",
  "model": "s3://ach-covid-lasso-us-east-2/inputs/model/lasso_model.joblib",
  "scaler": "s3://ach-covid-lasso-us-east-2/inputs/model/scaler.joblib",
  "outdir": "${OUTDIR}",
  "files": [{"filename":"tiny.fasta","content_type":"text/plain","size_bytes":50}]
}
EOF

echo "Running vegeta: rate=${RATE}/s duration=${DURATION}"
echo "POST ${API_BASE}/submit" | vegeta attack \
  -rate="$RATE" \
  -duration="$DURATION" \
  -header="Content-Type: application/json" \
  -header="x-api-key: ${API_KEY}" \
  -body="$BODY_FILE" \
  | tee /tmp/results.bin | vegeta report

echo
echo "Histogram:"
vegeta report -type=hist /tmp/results.bin
