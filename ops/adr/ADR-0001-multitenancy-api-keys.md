# ADR-0001: Multi-tenancy via API Keys + Tenant Scoping

## Status

✅ Accepted

## Context

The platform exposes a public control-plane endpoint (`/submit`) that creates jobs, presigns uploads, and can trigger compute. We need a lightweight multi-tenant model that:

* Enables per-tenant throttling and guardrails
* Prevents cross-tenant data visibility
* Keeps the public demo shareable while controlling blast radius

Constraints:

* Prefer native AWS primitives (API Gateway usage plans + DynamoDB) over a full auth system.
* Tenant identity must be available to Lambda for authorization decisions and for scoping S3 paths.

## Decision

* Use **API Gateway REST API keys** for access and map each request to a **tenant_id**.
* Resolve tenant identity via **`apiKeyId`** (API Gateway Key ID) from the request context, then look up tenant metadata in DynamoDB table **`covid_cfr_api_keys`**:

  * **PK:** `api_key_id` *(API Gateway Key ID — NOT the key value sent in `x-api-key`)*
  * Fields: `tenant_id`, `status`, optional per-tenant limits
* Stamp `tenant_id` onto each job row in **`covid_cfr_jobs`** (PK `job_id`).
* Enforce tenant-scoped storage via tenant-prefixed S3 prefixes:

  * Bucket: `ach-covid-lasso-us-east-2`
  * Uploads: `uploads/tenants/<tenant_id>/jobs/<job_id>/...`
  * Outputs: `results/tenants/<tenant_id>/jobs/<job_id>/...` *(or equivalent output prefix)*
* Enforce tenant isolation on reads/operations:

  * If a job exists but belongs to a different tenant, return **404** (do not leak job existence).

## Consequences

* **Pros**

  * Minimal auth surface area; works cleanly with API Gateway usage plans.
  * Per-tenant isolation is explicit and enforceable at the application layer.
  * Easy to add per-tenant caps and special-case admin/testing tenants.

* **Cons / tradeoffs**

  * API keys require rotation and careful secret handling (do not log key values).
  * Tenant resolution depends on API Gateway context wiring; misconfiguration can break auth.

* **Operational notes**

  * Expect common setup errors: confusing API key **ID** vs **value**. The key value is used in the `x-api-key` header; the key ID is stored in Dynamo.
  * Consider adding key lifecycle fields (`created_at`, `disabled_at`, `reason`) if multiple tenants are onboarded.

* **Follow-ups**

  * Optional: migrate to JWT/Cognito if richer auth/roles become necessary.
  * Optional: add per-tenant overrides (e.g., `max_pending_uploads`, `max_active_jobs`) in `covid_cfr_api_keys` for admin/testing tenants.
