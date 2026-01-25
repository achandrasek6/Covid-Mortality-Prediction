# Load Testing & Smoke Tests

This folder contains **safe-by-default** harnesses to validate the `/submit` control-plane and the end-to-end init → upload → finalize flow.

## What each test does

### 1) k6 init-only (recommended)

**File:** `k6_submit_init.js`
**Purpose:** measure control-plane latency + reliability under steady load without triggering AWS Batch compute.
**Does NOT:** upload to S3, run Batch, or finalize jobs.

Validates:

* API Gateway + submit Lambda hot path
* DynamoDB write path
* tenant lookup + guardrails behavior (429s)
* p95 latency under load

### 2) vegeta init-only (optional)

**File:** `vegeta_init.sh`
**Purpose:** quick throughput/latency sanity check (same init payload every request).
**Does NOT:** upload to S3 or run Batch.

### 3) E2E smoke (runs Batch once)

**File:** `smoke_e2e_upload_finalize.sh`
**Purpose:** correctness check for init → presign → upload → finalize, and finalize idempotency (dedupe).
**Does:** upload a tiny FASTA and calls finalize twice; should submit Batch exactly once.

---

## Prereqs

* `k6` installed (for load)
* `vegeta` installed (optional)
* `bash`, `curl`, `python3` (for smoke)

## Environment

```bash
export API_BASE="https://<restapi>.execute-api.<region>.amazonaws.com/<stage>"
export API_KEY="<api-key-value>"   # x-api-key VALUE (not key ID)
```

---

## Run: k6 init-only

### Baseline (2 req/s for 2m)

```bash
k6 run \
  -e API_BASE="$API_BASE" \
  -e API_KEY="$API_KEY" \
  -e RATE=2 \
  -e DURATION=2m \
  ops/loadtest/k6_submit_init.js
```

### Stress (10 req/s for 2m)

```bash
k6 run \
  -e API_BASE="$API_BASE" \
  -e API_KEY="$API_KEY" \
  -e RATE=10 \
  -e DURATION=2m \
  ops/loadtest/k6_submit_init.js
```

### Interpreting the k6 output (quick)

Prefer these custom metrics:

* `success_rate` (2xx fraction)
* `success_req_duration` p95 (latency for successes)
* `throttled` (429 fraction)
* `server_errors` (5xx fraction)
* `other_4xx` (4xx excluding 429)

---

## Run: vegeta init-only (optional)

```bash
export RATE=10
export DURATION=30s
bash ops/loadtest/vegeta_init.sh
```

If you want a histogram on the last run:

```bash
vegeta report -type=hist /tmp/results.bin
```

---

## Run: E2E smoke (runs Batch once)

```bash
bash ops/loadtest/smoke_e2e_upload_finalize.sh
```

Expected:

* prints `job_id=...`
* “uploaded tiny.fasta”
* first finalize returns “Job submitted” with `batch_job_id`
* second finalize returns “Job already submitted” (dedupe hit)
* job progresses `SUBMITTED → RUNNING → SUCCEEDED`

---

## Common gotcha: `TENANT_PENDING_LIMIT` during init-only load

Init-only tests can accumulate `pending_uploads` (since they don’t finalize). If you see:

`TENANT_PENDING_LIMIT: Too many pending uploads for tenant`

Use a high-limit admin tenant, or reset pending uploads between runs in dev.
