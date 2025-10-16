#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------
# run_nf_aws.sh — Submit Nextflow runner to AWS Batch & tail logs
# -------------------------------------------------------------
# Usage:
#   chmod +x run_nf_aws.sh
#   ./run_nf_aws.sh
#
# Optional: override defaults via env vars before running, e.g.:
#   REGION=us-east-2 QUEUE=nf-ec2 JOBDEF=covid-cfr-nf-runner:6 \
#   REFERENCE_FASTA=s3://.../NC_045512.2_sequence.fasta \
#   TRAIN_FEATURE_MATRIX=s3://.../feature_matrix_train.csv \
#   MODEL=s3://.../lasso_model.joblib \
#   SCALER=s3://.../scaler.joblib \
#   SAMPLES=s3://.../variant_samples_small.fasta \
#   OUTDIR=s3://.../results/ \
#   ./run_nf_aws.sh
# -------------------------------------------------------------

# --- Defaults (edit as needed) ---
: "${REGION:=us-east-2}"
: "${QUEUE:=nf-ec2}"
: "${JOBDEF:=covid-cfr-nf-runner:6}"

: "${REFERENCE_FASTA:=s3://ach-covid-lasso-us-east-2/inputs/reference/NC_045512.2_sequence.fasta}"
: "${TRAIN_FEATURE_MATRIX:=s3://ach-covid-lasso-us-east-2/inputs/lasso/feature_matrix_train.csv}"
: "${MODEL:=s3://ach-covid-lasso-us-east-2/inputs/model/lasso_model.joblib}"
: "${SCALER:=s3://ach-covid-lasso-us-east-2/inputs/model/scaler.joblib}"
: "${SAMPLES:=s3://ach-covid-lasso-us-east-2/inputs/test_samples/*.fasta}"
: "${OUTDIR:=s3://ach-covid-lasso-us-east-2/results/}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "Error: missing dependency '$1'" >&2; exit 1; }; }
need aws

submit_job() {
  echo "Submitting job to Batch…"
  local payload
  payload=$(cat <<JSON
{
  "environment": [
    {"name":"PROFILE","value":"aws"},
    {"name":"AWS_REGION","value":"${REGION}"},
    {"name":"VERBOSE","value":"true"},
    {"name":"REFERENCE_FASTA","value":"${REFERENCE_FASTA}"},
    {"name":"TRAIN_FEATURE_MATRIX","value":"${TRAIN_FEATURE_MATRIX}"},
    {"name":"MODEL","value":"${MODEL}"},
    {"name":"SCALER","value":"${SCALER}"},
    {"name":"SAMPLES","value":"${SAMPLES}"},
    {"name":"OUTDIR","value":"${OUTDIR}"}
  ],
  "command": []
}
JSON
)

  JOB_ID=$(aws batch submit-job \
    --region "$REGION" \
    --job-name "nf-runner-$(date +%Y%m%d-%H%M%S)" \
    --job-queue "$QUEUE" \
    --job-definition "$JOBDEF" \
    --container-overrides "$payload" \
    --query 'jobId' --output text)
  export JOB_ID
  echo "JOB_ID=$JOB_ID"
}

wait_for_stream() {
  echo "Waiting for CloudWatch log stream…"
  while true; do
    LOG_STREAM=$(aws batch describe-jobs --region "$REGION" --jobs "$JOB_ID" \
      --query 'jobs[0].attempts[-1].container.logStreamName' --output text 2>/dev/null || true)
    if [[ -n "$LOG_STREAM" && "$LOG_STREAM" != "None" ]]; then
      echo "Log stream: $LOG_STREAM"; export LOG_STREAM; return 0
    fi
    sleep 2
  done
}

tail_logs() {
  echo "Tailing logs… (Ctrl+C to stop)"
  aws logs tail "/aws/batch/job" --region "$REGION" --follow --log-stream-names "$LOG_STREAM"
}

job_status() {
  aws batch describe-jobs --region "$REGION" --jobs "$JOB_ID" \
    --query 'jobs[0].{Status:status,Reason:statusReason,Attempts:length(attempts),Queue:jobQueue,Def:jobDefinition}'
}

list_results() {
  echo "\nListing results in $OUTDIR …"
  aws s3 ls "$OUTDIR" --recursive --human-readable --summarize || true
}

main() {
  submit_job
  job_status
  wait_for_stream
  tail_logs || true
  echo "\nFinal job status:"; job_status
  list_results
}

main "$@"
