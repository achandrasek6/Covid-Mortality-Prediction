# ADR-0002: Batch State Changes via EventBridge → SQS (+ DLQ)

## Status

✅ Accepted

## Context

The UI and control plane require near-real-time job status updates. AWS Batch emits state changes (SUBMITTED/RUNNING/SUCCEEDED/FAILED) which must be applied to `covid_cfr_jobs` reliably.

Directly invoking a Lambda target from EventBridge can work, but has operational gaps:

* Limited buffering during downstream outages or deploys
* Harder to observe backlog/lag
* No built-in DLQ semantics for poison messages unless added explicitly

We want:

* Durable buffering and visibility into event lag
* Retriable processing with a DLQ
* Simple operational response (pause, redrive)

## Decision

* Route AWS Batch state change events through:

  **AWS Batch → EventBridge rule → SQS main queue → Lambda (`cfr-event-handler`) → DynamoDB (`covid_cfr_jobs`)**

* Configure SQS with:

  * Main queue: `cfr-batch-events-queue`

    * URL: `https://sqs.us-east-2.amazonaws.com/802861900950/cfr-batch-events-queue`
  * DLQ: `cfr-batch-events-dlq`

    * URL: `https://sqs.us-east-2.amazonaws.com/802861900950/cfr-batch-events-dlq`
  * Redrive policy (N receives) and visibility timeout aligned to handler runtime.

* Handler behavior:

  * Parse Batch event payload (jobId, status, timestamps)
  * Locate the corresponding job row via GSI `batch_job_id-index`
  * Apply state transitions to `covid_cfr_jobs`
  * On terminal states, perform one-time decrements of tenant counters (if enabled)
  * Emit structured logs for observability (`job_terminal_applied`, `record_processing_failed`)

* Alerting:

  * Page on DLQ non-empty
  * Monitor main queue age of oldest message and handler errors

## Consequences

* **Pros**

  * Reliability: buffering absorbs deploys/outages; retries are automatic.
  * Operability: queue depth/age gives clear signal of event lag.
  * Safety: DLQ isolates poison messages and enables controlled redrive.

* **Cons / tradeoffs**

  * Slight additional latency (SQS hop), typically acceptable for UI status.
  * Requires managing queue policies and permissions.

* **Operational notes**

  * DLQ non-empty is a high-signal page; first action is to inspect a sample message and correlate with handler logs.
  * During systemic failures, disable the SQS → Lambda event source mapping to stop churn, then re-enable after fix.

* **Follow-ups**

  * Consider idempotency/deduping in the handler if event replay becomes common (e.g., last-applied timestamp).
  * Add a metric for end-to-end event application lag (Batch timestamp → Dynamo update time).
