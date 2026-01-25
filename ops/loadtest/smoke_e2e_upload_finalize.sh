#!/usr/bin/env bash
set -euo pipefail

: "${API_BASE:?set API_BASE}"
: "${API_KEY:?set API_KEY}"

printf ">x\nACGT\n" > tiny.fasta

INIT_JSON="$(curl -sS -X POST "$API_BASE/submit" \
  -H "content-type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "phase":"init",
    "reference_fasta":"s3://ach-covid-lasso-us-east-2/inputs/reference/NC_045512.2_sequence.fasta",
    "train_feature_matrix":"s3://ach-covid-lasso-us-east-2/inputs/lasso/feature_matrix_train.csv",
    "model":"s3://ach-covid-lasso-us-east-2/inputs/model/lasso_model.joblib",
    "scaler":"s3://ach-covid-lasso-us-east-2/inputs/model/scaler.joblib",
    "outdir":"s3://ach-covid-lasso-us-east-2/results",
    "files":[{"filename":"tiny.fasta","content_type":"text/plain","size_bytes":50}]
  }')"

JOB_ID="$(echo "$INIT_JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin)["job_id"])')"
POST_URL="$(echo "$INIT_JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin)["files"][0]["post"]["url"])')"
FIELDS="$(echo "$INIT_JSON" | python3 -c 'import sys,json; import json as j; print(j.dumps(json.load(sys.stdin)["files"][0]["post"]["fields"]))')"

echo "job_id=$JOB_ID"

# Build curl args safely (no word-splitting issues)
mapfile -t FIELD_LINES < <(python3 - "$FIELDS" <<'PY'
import json, sys
fields = json.loads(sys.argv[1])
for k, v in fields.items():
    print(f"{k}={v}")
PY
)

CURL_ARGS=()
for kv in "${FIELD_LINES[@]}"; do
  CURL_ARGS+=(-F "$kv")
done
CURL_ARGS+=(-F "file=@tiny.fasta;type=text/plain")

curl -sS -X POST "$POST_URL" "${CURL_ARGS[@]}" >/dev/null
echo "uploaded tiny.fasta"

# Call finalize twice (tests dedupe/lock behavior)
curl -sS -X POST "$API_BASE/submit" -H "content-type: application/json" -H "x-api-key: $API_KEY" -d "{\"phase\":\"finalize\",\"job_id\":\"$JOB_ID\"}" &
curl -sS -X POST "$API_BASE/submit" -H "content-type: application/json" -H "x-api-key: $API_KEY" -d "{\"phase\":\"finalize\",\"job_id\":\"$JOB_ID\"}" &
wait

echo "finalize called twice"
