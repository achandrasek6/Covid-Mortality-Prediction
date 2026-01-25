# ADR-0003: Tenant Counters + Guardrails (Pending Uploads / Active Jobs)

## Status

✅ Accepted

## Context

The `/submit` endpoint is public-facing and can be abused (intentional or accidental), leading to:

* Unbounded job creation and storage growth
* Compute spend spikes (Batch submissions)
* Degraded latency/availability for all users

We need tenant-level guardrails that:

* Prevent one tenant from exhausting shared resources
* Are cheap and atomic under concurrency
* Work with both init-only job creation and compute submission

## Decision

* Enforce **per-tenant concurrency caps** using counters:

  * `pending_uploads`: jobs created by `phase=init` that have not been finalized
  * `active_jobs`: jobs that have submitted compute and are not terminal

* Maintain counters in DynamoDB table **`tenant_counters`** (PK `tenant_id`) *(if enabled)*.

  * Update counters using **atomic ADD** and **conditional expressions** to enforce caps.

* Guardrail behavior:

  * On `phase=init`:

    * Validate tenant is active
    * Conditionally increment `pending_uploads` (reject if at limit)
    * Create the job row and return presigned upload POST
  * On `phase=finalize` (winner request):

    * Transition job status via a conditional lock (see ADR-0004)
    * Decrement `pending_uploads` exactly once
    * Increment `active_jobs` exactly once
    * Submit Batch compute
  * On terminal Batch states (handler):

    * Decrement `active_jobs` exactly once

* Error semantics:

  * If `pending_uploads` cap is exceeded: return 429 with `error_code=TENANT_PENDING_LIMIT`
  * If `active_jobs` cap is exceeded: return 429 with `error_code=TENANT_ACTIVE_LIMIT`

* Admin/testing tenants:

  * Support higher caps via per-tenant overrides in `covid_cfr_api_keys` (e.g., `max_pending_uploads`, `max_active_jobs`) with environment defaults as fallback.

## Consequences

* **Pros**

  * Hard cap on blast radius per tenant; predictable cost control.
  * Counters are cheap (single Dynamo update) and safe under concurrency.
  * Enables transparent throttling semantics (429) instead of 5xx.

* **Cons / tradeoffs**

  * Requires careful “exactly-once” updates to avoid counter drift.
  * Init-only tests can accumulate `pending_uploads` if finalize is not called (expected).

* **Operational notes**

  * If many requests hit `TENANT_PENDING_LIMIT`, either the tenant is at cap (expected) or counters are stuck high.
  * For dev/testing, counters may be reset between runs; for production, prefer TTL/sweeper for abandoned uploads.

* **Follow-ups**

  * Add TTL to `PENDING_UPLOAD` job rows and a sweeper to decrement abandoned `pending_uploads`.
  * Consider per-tenant rate limits in addition to concurrency caps if needed.
