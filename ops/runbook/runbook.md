# COVID CFR Platform — Operations Runbook

This runbook covers the core control-plane (`/submit`) and the event-driven status pipeline (Batch → EventBridge → SQS → Lambda → DynamoDB). It is intentionally short and action-oriented.

## Services

* **submit-cfr-job** (API Gateway → Lambda)

  * `phase=init`: creates job row, validates tenant, increments pending uploads (guardrail), presigns S3 uploads
  * `phase=finalize`: validates uploads, transitions state with conditional lock, submits Batch exactly once, updates counters

* **cfr-event-handler** (SQS → Lambda)

  * consumes Batch state-change events from the main queue
  * applies status transitions to DynamoDB (and decrements counters on terminal)

* **Event pipeline**

  * Batch → EventBridge rule → **SQS main** → Lambda → DynamoDB
  * failures → **SQS DLQ** (paged via SNS)

## Key resources

### DynamoDB tables

* **covid_cfr_jobs**

  * PK: `job_id`
  * GSI: `batch_job_id-index` (for event handler lookups)
  * Typical fields: `tenant_id`, `status`, `batch_job_id`, timestamps, `error_code`, `download_url` / outputs

* **covid_cfr_api_keys**

  * PK: `api_key_id` (API Gateway Key ID)
  * Fields: `tenant_id`, `status`, per-tenant limits (optional)

* **tenant_counters** (if enabled)

  * PK: `tenant_id`
  * Fields: `pending_uploads`, `active_jobs`, optional max overrides

### SQS

* **cfr-batch-events-queue** (main)
* **cfr-batch-events-dlq** (DLQ)

### AWS Batch

* Job queue(s): e.g. `nf-ec2` (varies by env)
* Compute environment(s): (varies by env)

## Golden signals

### Submit path

* Request rate, **5xx rate**, **p95 latency**
* 429 rate (tenant caps / usage plan throttles)

### Event path

* Main queue depth + age of oldest message
* Handler Lambda errors
* DLQ non-empty

### Correctness

* Jobs stuck in `PENDING_UPLOAD` or `SUBMITTING`
* Batch jobs in `RUNNING` but UI/job row not updating

## Dashboards / Logs (examples)

CloudWatch log groups:
- `/aws/lambda/submit-cfr-job`
- `/aws/lambda/cfr-event-handler`

The Logs Insights queries below assume structured fields like `msg`, `status_code`, `error_code`, `tenant_id`, `duration_ms`, `phase`, `status`, `job_id`, and `batch_job_id`.

### Saved queries

**Per-tenant response summary (p50/p95 + 5xx count)**

```sql
fields @timestamp, tenant_id, msg, status_code, duration_ms, error_code
| filter msg="submit_response"
| stats
    count() as n,
    pct(duration_ms, 50) as p50_ms,
    pct(duration_ms, 95) as p95_ms,
    sum(status_code>=500) as n_5xx
  by tenant_id
```
**Trace a single job end-to-end (submit + handler)**
```sql
fields @timestamp, @log, msg, phase, status, error_code, duration_ms, job_id, batch_job_id
| filter job_id="<JOB_ID>"
| sort @timestamp asc
```

### Submit Lambda

* Count responses by status code:

  * `filter msg="submit_response" | stats count() by status_code`
* Guardrail hits:

  * `filter error_code="TENANT_PENDING_LIMIT" or error_code="TENANT_ACTIVE_LIMIT" | stats count() by tenant_id`

### Handler Lambda

* Terminal transitions applied:

  * `filter msg="job_terminal_applied" | stats count() by tenant_id`
* Event handler failures:

  * `filter msg="record_processing_failed" | stats count() by error_code`

## Alerts

These CloudWatch alarms are configured for the critical control-plane and event pipeline.

| Alarm name            | Signal | Condition (5m window) | Primary action |
|-----------------------|---|---|---|
| `cfr-dlq-nonempty`    | SQS DLQ has messages | `ApproximateNumberOfMessagesVisible >= 1` (1 datapoint) | Follow **DLQ non-empty** playbook; inspect one DLQ message, correlate handler logs, then redrive if safe |
| `submission_failure`  | Submit Lambda errors | `SubmitErrorCount >= 1` (1 datapoint) | Check `/aws/lambda/submit-cfr-job` logs for recent errors; validate API key mapping + Dynamo write path |
| `batch_submit_failed` | Finalize failed to submit Batch | `BatchSubmitFailed >= 1` (1 datapoint) | Check submit Lambda finalize logs; confirm Batch job queue `nf-ec2` and permissions; review any `error_code` persisted on job row |
| `status=error`        | Event handler errors | `HandlerErrorCount >= 1` (1 datapoint) | Check `/aws/lambda/cfr-event-handler` logs; confirm main queue depth/inflight; watch for parsing/GSI lookup failures |

Notes:
- **DLQ non-empty** is a high-signal page; treat it as urgent until cleared.
- “Insufficient data” is normal when traffic is low; alarms become meaningful once metrics emit.


---

# Triage playbooks

## 1) DLQ non-empty

**Goal:** determine whether failures are transient (safe to redrive) vs systemic (pause consumption).

1. Check main queue backlog + inflight:

   ```bash
   aws sqs get-queue-attributes \
     --queue-url "$MAIN_URL" \
     --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible
   ```
2. Sample one DLQ message:

   ```bash
   aws sqs receive-message --queue-url "$DLQ_URL" --max-number-of-messages 1
   ```
3. Correlate timestamp with handler logs (look for parsing errors, missing GSI rows, permissions).
4. If the issue was transient and is fixed, **redrive** DLQ → main queue.
5. If systemic, **disable** the event source mapping to stop churn (see Safe toggles).

## 2) Jobs stuck in `PENDING_UPLOAD`

**Likely causes:** upload never happened, client abandoned, or the presigned POST failed.

Steps:

1. Inspect the job row in `covid_cfr_jobs`: status, upload prefix, created_at.
2. Verify objects exist under the job’s upload prefix in S3.
3. If uploads are now present, re-run `phase=finalize` (idempotent).
4. If abandoned, consider TTL/sweeper cleanup in dev.

## 3) Jobs stuck in `SUBMITTING`

**Likely causes:** winner finalize crashed after locking but before persisting `batch_job_id`, or Batch submit failed.

Steps:

1. Inspect job row for `batch_job_id` and `error_code`.
2. Check submit Lambda logs around finalize for the job_id.
3. If safe in your implementation, re-run finalize (should return “already in progress” or resubmit based on lock/lease semantics).
4. If you use a lease/timeout on SUBMITTING, verify it’s behaving as intended.

## 4) UI not updating job status

**Likely causes:** Batch events not reaching SQS, mapping disabled, handler failing, GSI query issues.

Steps:

1. Confirm Batch job exists and has a status in the Batch console.
2. Confirm EventBridge rule is matching Batch events.
3. Confirm messages are arriving in the main SQS queue.
4. Confirm event source mapping is enabled and handler is processing (CloudWatch errors = 0).
5. Confirm handler can query `batch_job_id-index` and update `covid_cfr_jobs`.

## 5) High rate of `TENANT_PENDING_LIMIT` / `TENANT_ACTIVE_LIMIT`

**Interpretation:** guardrails are working, but limits may be too strict for current usage.

Steps:

1. Confirm whether the spike is one tenant or many tenants.
2. For admin/testing tenants, raise per-tenant limits or reset counters in dev.
3. If production, consider usage-plan throttles + per-tenant overrides + messaging to users.

---

# Safe toggles / emergency actions

## Disable SQS → Lambda consumption (pause handler)

```bash
aws lambda update-event-source-mapping --uuid <UUID> --no-enabled
```

## Re-enable consumption

```bash
aws lambda update-event-source-mapping --uuid <UUID> --enabled
```

## Redrive DLQ → main queue

Use the console redrive UI, or the CLI (exact command varies by setup). Prefer console for safety unless automated.

---

# Post-incident checklist

* Identify root cause (throttle, permissions, schema change, missing index, parsing bug)
* Add/adjust alarms (DLQ, handler errors, queue age)
* Add a regression test (smoke or unit) to prevent recurrence
* Update ADRs/runbook if the operational model changed
