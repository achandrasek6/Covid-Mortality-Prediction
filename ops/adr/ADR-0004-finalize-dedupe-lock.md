# ADR-0004: Finalize Idempotency via Conditional Transition Lock

## Status

✅ Accepted

## Context

Clients can retry finalize (network failures, double-clicks), and the UI may issue duplicate requests. Without protection, a single job could submit AWS Batch compute multiple times, causing:

* Duplicate cost
* Confusing status state
* Counter drift (`active_jobs` increments twice)

We need finalize to be:

* **Idempotent** (safe to call multiple times)
* **At-most-once** for Batch submission per `job_id`
* Clear on retry semantics (in progress vs already submitted)

## Decision

* Implement a DynamoDB conditional update as a lock on the job row:

  * Only allow transition **`PENDING_UPLOAD → SUBMITTING`** if current status is `PENDING_UPLOAD`.

* Semantics:

  * If the conditional update succeeds: this request is the **winner** and may submit Batch.
  * If the conditional update fails because status is already `SUBMITTING`/`SUBMITTED`/`RUNNING`/terminal:

    * Return a dedupe response indicating the job is already in progress or complete.

* Persist `batch_job_id` on the job row for correlation and for idempotent responses.

* Suggested responses:

  * **200** for dedupe hit (“already submitted”) including `batch_job_id`
  * **202** for “in progress” (optional) where clients should poll status

* Counter correctness:

  * Apply counter transitions only on the winner path:

    * decrement `pending_uploads` exactly once
    * increment `active_jobs` exactly once
  * Apply terminal decrements exactly once in the event handler.

## Evidence / validation

An end-to-end smoke test validated the behavior:

* Finalize #1 returned **“Job submitted”** with a `batch_job_id`
* Finalize #2 returned **“Job already submitted”** with the **same** `batch_job_id`
* The job progressed through expected statuses: `SUBMITTED → RUNNING → SUCCEEDED`

## Consequences

* **Pros**

  * Guarantees at-most-once Batch submission per `job_id` under concurrent finalize calls.
  * Makes retries safe and predictable for clients.

* **Cons / tradeoffs**

  * If the winner crashes after acquiring the lock but before persisting `batch_job_id`, jobs can be stuck in `SUBMITTING`.

* **Operational notes**

  * If jobs are stuck in `SUBMITTING`, check submit Lambda logs for the finalize winner and decide on a safe retry policy.
  * Consider a lease/timeout on `SUBMITTING` if this becomes common.

* **Follow-ups**

  * Add a short lease to `SUBMITTING` (e.g., allow retry if `updated_at` is older than N minutes).
  * Add explicit metrics: dedupe-hit rate, finalize winner rate, and submit-to-terminal time.
