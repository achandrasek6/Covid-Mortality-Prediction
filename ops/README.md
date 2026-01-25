# Ops (Load Testing, Runbooks, ADRs)

This folder contains operational artifacts for the COVID CFR platform: safe load tests, end-to-end smoke checks, on-call runbooks, and architecture decision records (ADRs). The goal is to make reliability, guardrails, and operational behavior auditable and easy to reason about.

## Directory layout

```
ops/
  README.md
  loadtest/
    README.md
    k6_submit_init.js
    vegeta_init.sh
    smoke_e2e_upload_finalize.sh
  runbook/
    RUNBOOK.md
  adr/
    ADR-0001-multitenancy-api-keys.md
    ADR-0002-batch-events-sqs-dlq.md
    ADR-0003-tenant-counters-guardrails.md
    ADR-0004-finalize-dedupe-lock.md
```

## What lives here

* **Load testing (`ops/loadtest/`)**: cheap-by-default tests that validate the `/submit` control-plane hot path and capture latency/throttle behavior.
* **E2E smoke (`ops/loadtest/smoke_e2e_upload_finalize.sh`)**: correctness check for init → upload → finalize (includes finalize idempotency).
* **Runbook (`ops/runbook/`)**: triage steps and safe toggles for the submit path and the Batch event pipeline.
* **ADRs (`ops/adr/`)**: short “why we chose this” docs for decisions that affect reliability and ops.

## Quickstart

### Prereqs

* `k6` installed (for load)
* `bash`, `curl`, `python3` (for smoke)
* An API key (send as `x-api-key`) and an API base URL

### Environment

```bash
export API_BASE="https://<restapi>.execute-api.<region>.amazonaws.com/<stage>"
export API_KEY="<api-key-value>"
```

## Load testing: control-plane init (safe, no Batch)

`phase=init` creates a job row and presigns uploads. It does **not** upload data and does **not** submit AWS Batch compute.

### k6 (recommended)

```bash
k6 run \
  -e API_BASE="$API_BASE" \
  -e API_KEY="$API_KEY" \
  -e RATE=2 \
  -e DURATION=2m \
  ops/loadtest/k6_submit_init.js
```

#### Recorded baselines

* **2 req/s for 2m**: p95 ≈ **217.6 ms**, **0%** server errors, **0%** throttling (n=241)
* **10 req/s for 2m**: p95 ≈ **223.2 ms**, **0%** server errors, **0%** throttling (n=1198)

> Note: init-only testing can hit the tenant `pending_uploads` guardrail if pending uploads are not cleaned up. Use a high-limit admin tenant for load tests, or reset pending uploads between runs.

### vegeta (optional)

Vegeta is a lightweight alternative to k6 for quick throughput/latency sanity checks.

```bash
# example
export RATE=10
export DURATION=30s
bash ops/loadtest/vegeta_init.sh
```

#### Recorded baseline (dev)

* 10 req/s for 30s: **300 requests**, **100% success (200:300)**
* Latency: p50 ≈ **191ms**, p95 ≈ **246ms**, p99 ≈ **1.22s**, max ≈ **1.28s**

> Note: this repo uses `vegeta report -type=hist` for histogram output.

## End-to-end smoke (runs Batch once)

This is a correctness smoke, not a load test. It performs init → upload → finalize and calls finalize twice to validate idempotency/deduping. This will typically submit **one** Batch job.

```bash
bash ops/loadtest/smoke_e2e_upload_finalize.sh
```

### Recorded run (dev)

* Example run `job_id`: `8a5385a4-4514-42ef-aa1f-9072a31426bd`
* Upload: **success** (presigned POST accepted)
* Finalize #1: **Job submitted**, `batch_job_id=310d3f27-c89d-4a6c-8a7a-59fa215a196a`, status `SUBMITTED`
* Finalize #2: **Job already submitted** (dedupe hit), same `batch_job_id`, status `SUBMITTED`

This validates finalize idempotency (at-most-once Batch submission per job_id) and end-to-end correctness for init/presign/upload/finalize.

### Status propagation (validated)

The submitted job progressed through the expected lifecycle and reached a terminal success state, with status updates reflected correctly end-to-end (submit → event pipeline → Dynamo/UI):

* `SUBMITTED` → `RUNNING` → `SUCCEEDED`

> Intermediate states (e.g., PENDING, RUNNABLE) omitted for brevity.

## Runbook

See `ops/runbook/RUNBOOK.md` for triage playbooks and safe toggles.

## ADRs

See `ops/adr/` for architecture decisions impacting reliability and operations.

- **ADR-0001: Multi-tenancy via API keys + tenant scoping** — Map API Gateway `apiKeyId` → `tenant_id` in Dynamo, stamp tenant on jobs, and enforce tenant-scoped S3 prefixes and isolation.
- **ADR-0002: Batch state changes via EventBridge → SQS (+ DLQ)** — Buffer Batch events in SQS for retries/visibility; route poison messages to DLQ and page on DLQ non-empty.
- **ADR-0003: Tenant counters + guardrails** — Concurrency caps (`pending_uploads`, `active_jobs`) enforced via atomic Dynamo updates to prevent abuse and contain blast radius.
- **ADR-0004: Finalize dedupe/lock semantics** — Conditional state transition lock to guarantee at-most-once Batch submission per `job_id` and make finalize idempotent.

## SLOs and guardrails

**Control-plane SLOs (target + validated baselines).**

* Availability (control plane): **≥99.5%** successful responses where **2xx/expected 4xx** are considered non-outages (exclude auth failures).
* Latency (control plane): **p95 < 800ms** for `/submit` **phase=init** and **phase=finalize** under steady load.

**Validated baselines (dev).**

* `/submit` **phase=init** @ **2 req/s for 2m**: p95 ≈ **217.6ms**, 0% server errors, 0% throttling (n=241)
* `/submit` **phase=init** @ **10 req/s for 2m**: p95 ≈ **223.2ms**, 0% server errors, 0% throttling (n=1198)

**Multi-tenant isolation / blast-radius containment.**

* API-key → tenant mapping (Dynamo) with tenant_id stamped on job rows.
* Tenant-scoped S3 prefixes for uploads/outputs.
* Per-tenant concurrency caps (pending uploads + active jobs) to prevent abuse.

**Event propagation (compute-plane status).**

* Batch state changes are event-driven (EventBridge → SQS → Lambda → DynamoDB) with DLQ paging.


## Interpreting load test metrics

The k6 harness reports both built-in HTTP timings and custom metrics so expected throttling doesn’t get confused with reliability issues.

### Custom metrics (preferred)

* **`success_rate`**: fraction of requests that returned **2xx**.
* **`success_req_duration`**: latency (ms) for **2xx responses only**. Use p95/p99 here for SLO tail latency.
* **`throttled`**: fraction of requests that returned **429** (tenant cap or upstream throttling). High throttled rate is expected if you intentionally overdrive.
* **`server_errors`**: fraction of requests that returned **5xx**. This should be ~0 under normal load.
* **`other_4xx`**: fraction of **4xx excluding 429**. Should be near zero (usually indicates a client bug or a contract mismatch).

### Built-in metrics (use with care)

* **`http_req_duration`**: client-observed end-to-end latency for all responses (includes 429s and internet noise). Useful for overall shape; prefer `success_req_duration` for SLO.
* **`http_req_failed`**: in k6 this is not “network only”; it can be misleading because it treats non-2xx as failures. Prefer the custom metrics above.

## Reset / cleanup notes for init-only load tests

Init-only tests (`phase=init`) can accumulate tenant state depending on how guardrails are implemented.

### Symptom: `TENANT_PENDING_LIMIT`

If you see errors like:

* `{"error":"Too many pending uploads for tenant","error_code":"TENANT_PENDING_LIMIT"}`

…then the tenant’s `pending_uploads` counter is at its cap. This is expected if you only run init (which increments pending) without finalize (which decrements).

### Recommended approaches

* **Use a high-limit admin tenant** for load testing (preferred).
* **Reset the admin tenant counters** between runs in dev (acceptable for testing).
* Add a **cleanup policy** for abandoned uploads (TTL + sweeper) if this becomes common in real usage.

### Optional dev reset (if you maintain a tenant counters table)

If you have a `tenant_counters` DynamoDB table, you can reset a tenant in dev between runs:

```bash
aws dynamodb update-item \
  --table-name tenant_counters \
  --key '{"tenant_id":{"S":"<tenant_id>"}}' \
  --update-expression "SET pending_uploads = :z" \
  --expression-attribute-values '{":z":{"N":"0"}}'
```

## Links

* **Load testing**: `ops/loadtest/README.md` — how to run k6/vegeta and the E2E smoke.
* **Runbook**: `ops/runbook/RUNBOOK.md` — triage playbooks (DLQ non-empty, stuck jobs, missing status updates) and safe toggles.
* **ADRs**: `ops/adr/` — short decision records:

  * ADR-0001: API keys → tenant mapping + tenant scoping
  * ADR-0002: Batch events via SQS + DLQ
  * ADR-0003: Tenant counters + guardrails
  * ADR-0004: Finalize dedupe/lock semantics
