// src/App.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

const API_BASE =
  import.meta.env.VITE_API_BASE ||
  "https://dhjv2dg2l1.execute-api.us-east-2.amazonaws.com/dev";

type JobStatus = {
  job_id: string;
  status: string;
  created_at?: string;
  updated_at?: string;
  batch_job_id?: string;
  error_message?: string | null;
  download_url?: string;
};

type JobMeta = {
  name?: string;
  description?: string;
};

type SubmitPayload = {
  samples_uri: string;
  reference_fasta: string;
  train_feature_matrix: string;
  model: string;
  scaler: string;
  outdir: string;
};

type ModalContent = {
  title: string;
  body: string;
};

type ConfirmContent = {
  title: string;
  body: string;
  confirmLabel?: string;
  cancelLabel?: string;
  danger?: boolean;
  onConfirm: () => void;
};

const JOB_NAME_MAX = 60;
const JOB_DESC_MAX = 500;
const URI_MAX = 260;

// Demo behavior: allow the checkbox + revealed field, but lock the revealed custom S3 inputs.
const DEMO_LOCK_CUSTOM_S3 = true;

const SAMPLE_OPTIONS = [
  {
    label: "Quick demo (tiny): 10 sequences (7 valid + 3 random reject)",
    value:
      "s3://ach-covid-lasso-us-east-2/inputs/test_samples/variant_samples_tiny.fasta",
    hint:
      "Fastest option. Runs 10 sequences total: 7 should align and produce predictions; 3 are random DNA and are expected to fail alignment and be routed to the rejected output.",
  },
  {
    label: "Demo (single-file input): small COVID genome FASTA",
    value:
      "s3://ach-covid-lasso-us-east-2/inputs/test_samples/variant_samples_small.fasta",
    hint:
      "Single-file demo: one FASTA containing a small set of SARS-CoV-2 genomes.",
  },
  {
    label: "Demo (multi-file input): demo FASTA + reject test",
    value: "s3://ach-covid-lasso-us-east-2/inputs/test_samples/*",
    hint:
      "Multi-file demo: includes (1) A small SARS-CoV-2 variant samples file and (2) a reject test file where some records intentionally fail validation and are routed to a rejected output.",
  },
];

const REFERENCE_OPTIONS = [
  {
    label: "Reference genome: Wuhan (NC_045512.2)",
    value:
      "s3://ach-covid-lasso-us-east-2/inputs/reference/NC_045512.2_sequence.fasta",
  },
];

const FEATURE_MATRIX_OPTIONS = [
  {
    label: "Training features (Lasso) — fixed artifact",
    value:
      "s3://ach-covid-lasso-us-east-2/inputs/lasso/feature_matrix_train.csv",
  },
];

const MODEL_OPTIONS = [
  {
    label: "Prediction model (Lasso CFR) — fixed artifact",
    value: "s3://ach-covid-lasso-us-east-2/inputs/model/lasso_model.joblib",
  },
];

const SCALER_OPTIONS = [
  {
    label: "Feature scaler — fixed artifact",
    value: "s3://ach-covid-lasso-us-east-2/inputs/model/scaler.joblib",
  },
];

const OUTDIR_OPTIONS = [
  {
    label: "Results folder (S3)",
    value: "s3://ach-covid-lasso-us-east-2/results/",
  },
];

// ---- Bundled model package (single dropdown) ----
type ModelPackage = {
  label: string;
  reference_fasta: string;
  train_feature_matrix: string;
  model: string;
  scaler: string;
};

const MODEL_PACKAGE_OPTIONS = [
  {
    label: "Lasso CFR (v1) — Wuhan ref + fixed features + scaler",
    value: "lasso_v1",
  },
] as const;

// ✅ IMPORTANT: derive a union type from the option values
type ModelPackageChoice = (typeof MODEL_PACKAGE_OPTIONS)[number]["value"];

// ✅ Guard to safely accept persisted strings / <select> values
const isModelPackageChoice = (v: unknown): v is ModelPackageChoice =>
  typeof v === "string" && MODEL_PACKAGE_OPTIONS.some((o) => o.value === v);

const MODEL_PACKAGES: Record<ModelPackageChoice, ModelPackage> = {
  lasso_v1: {
    label: "Lasso CFR (v1) — Wuhan ref + fixed features + scaler",
    reference_fasta: REFERENCE_OPTIONS[0].value,
    train_feature_matrix: FEATURE_MATRIX_OPTIONS[0].value,
    model: MODEL_OPTIONS[0].value,
    scaler: SCALER_OPTIONS[0].value,
  },
};

// ---- Fixed advanced settings (UI-only) ----
const ADVANCED_DEFAULTS = {
  min_alignment_identity: 0.92,
  chunk_size_samples: 10, // formerly "branching"
  max_branches: 50,
};

function normalizeStatus(status?: string) {
  const s = (status || "UNKNOWN").toUpperCase();
  if (s === "PENDING") return "RUNNABLE";
  return s;
}

function statusColors(status?: string) {
  const s = normalizeStatus(status);
  const palette: Record<string, { bg: string; border: string; text: string }> =
    {
      SUBMITTED: { bg: "#f1f5f9", border: "#cbd5e1", text: "#334155" },
      RUNNABLE: { bg: "#eff6ff", border: "#bfdbfe", text: "#1d4ed8" },
      STARTING: { bg: "#faf5ff", border: "#e9d5ff", text: "#6d28d9" },
      RUNNING: { bg: "#fffbeb", border: "#fde68a", text: "#b45309" },
      SUCCEEDED: { bg: "#ecfdf5", border: "#a7f3d0", text: "#047857" },
      FAILED: { bg: "#fff1f2", border: "#fecdd3", text: "#be123c" },
      DONE: { bg: "#f8fafc", border: "#e2e8f0", text: "#475569" },
    };
  return palette[s] || { bg: "#f8fafc", border: "#e2e8f0", text: "#475569" };
}

const STATUS_HELP: Record<string, string> = {
  SUBMITTED: "Request accepted. Waiting to enter the queue.",
  RUNNABLE: "Queued and waiting for compute capacity.",
  STARTING: "Compute allocated. Job is starting up.",
  RUNNING: "Job is actively running.",
  SUCCEEDED: "Completed successfully. Download is available.",
  FAILED:
    "Completed with an error. Open Job details to see the error message (if available).",
  DONE: "Terminal step. ✓ if succeeded, ⚠ if failed.",
  UNKNOWN: "Status is not recognized.",
};

function statusIconPrefix(status?: string) {
  const s = normalizeStatus(status);
  if (s === "SUCCEEDED") return "✓ ";
  if (s === "FAILED") return "⚠ ";
  return "";
}

function formatTime(ts?: string) {
  if (!ts) return "-";
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
}

function formatTimeCompact(ts?: string) {
  if (!ts) return "-";
  try {
    return new Date(ts).toLocaleString(undefined, {
      year: "2-digit",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return ts;
  }
}

function relativeTime(ts?: string, nowMs: number = Date.now()) {
  if (!ts) return "";
  const ms = Date.parse(ts);
  if (!Number.isFinite(ms)) return "";
  const diffSec = Math.floor((nowMs - ms) / 1000);
  if (diffSec < 0) return "just now";
  if (diffSec < 10) return "just now";
  if (diffSec < 60) return `${diffSec}s ago`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  return `${diffDay}d ago`;
}

function truncateId(id: string, left = 8, right = 6) {
  if (!id) return "";
  if (id.length <= left + right + 3) return id;
  return `${id.slice(0, left)}…${id.slice(-right)}`;
}

function useWindowWidth() {
  const [w, setW] = useState(() => window.innerWidth);
  useEffect(() => {
    const onResize = () => setW(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  return w;
}

async function fetchJobStatus(jobId: string): Promise<JobStatus> {
  const resp = await fetch(`${API_BASE}/status/${encodeURIComponent(jobId)}`);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data?.error || "Failed to fetch job status");
  return data as JobStatus;
}

function parseTimeMs(ts?: string): number | null {
  if (!ts) return null;
  const ms = Date.parse(ts);
  return Number.isFinite(ms) ? ms : null;
}

function createdMs(job: JobStatus) {
  return parseTimeMs(job.created_at) ?? Number.NEGATIVE_INFINITY;
}

function updatedMs(job: JobStatus) {
  return (
    parseTimeMs(job.updated_at) ??
    parseTimeMs(job.created_at) ??
    Number.NEGATIVE_INFINITY
  );
}

/**
 * Unified hover tooltip for the whole app:
 * - Black text on white background (matches the rest of the UI)
 * - Subtle border + shadow
 * - Supports block wrapping to avoid shrink-wrapping full-width inputs
 */
function HoverTip({
  text,
  children,
  maxWidth = 340,
  block = false,
}: {
  text: string;
  children: React.ReactNode;
  maxWidth?: number;
  block?: boolean;
}) {
  const anchorRef = useRef<HTMLSpanElement | null>(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);

  const show = () => {
    const el = anchorRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();

    const padding = 10;
    let left = r.left;
    left = Math.min(left, window.innerWidth - maxWidth - padding);
    left = Math.max(padding, left);

    const top = Math.min(r.bottom + 8, window.innerHeight - 16);
    setPos({ top, left });
    setOpen(true);
  };

  const hide = () => setOpen(false);

  return (
    <span
      ref={anchorRef}
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
      style={{
        display: block ? "block" : "inline-flex",
        width: block ? "100%" : undefined,
        alignItems: "center",
        minWidth: 0,
      }}
    >
      {children}

      {open && pos
        ? createPortal(
            <div
              role="tooltip"
              style={{
                position: "fixed",
                top: pos.top,
                left: pos.left,
                zIndex: 2000,
                maxWidth,
                background: "#ffffff",
                color: "#0f172a",
                border: "1px solid #e2e8f0",
                borderRadius: 12,
                padding: "8px 10px",
                fontSize: 11,
                lineHeight: 1.35,
                boxShadow:
                  "0 16px 40px rgba(15,23,42,0.12), 0 2px 6px rgba(15,23,42,0.08)",
              }}
            >
              {text}
            </div>,
            document.body
          )
        : null}
    </span>
  );
}

// Convenience wrapper: if text is empty/undefined, do not wrap.
function TipWrap({
  text,
  children,
  block = false,
  maxWidth,
}: {
  text?: string | null;
  children: React.ReactNode;
  block?: boolean;
  maxWidth?: number;
}) {
  if (!text) return <>{children}</>;
  return (
    <HoverTip text={text} block={block} maxWidth={maxWidth}>
      {children}
    </HoverTip>
  );
}

function StatusPill({ status }: { status?: string }) {
  const s = normalizeStatus(status);
  const c = statusColors(s);
  const tooltip = STATUS_HELP[s] || STATUS_HELP.UNKNOWN;

  return (
    <HoverTip text={tooltip}>
      <span
        style={{
          display: "inline-flex",
          alignItems: "center",
          borderRadius: 999,
          padding: "3px 8px",
          fontSize: 10,
          fontWeight: 800,
          background: c.bg,
          border: `1px solid ${c.border}`,
          color: c.text,
          letterSpacing: 0.2,
          whiteSpace: "nowrap",
        }}
        aria-label={`Status: ${s}`}
      >
        {statusIconPrefix(s)}
        {s}
      </span>
    </HoverTip>
  );
}

/**
 * Status pill that ALSO acts as the lifecycle toggle (caret integrated).
 * Keeps the "low profile" ✓ / ⚠ prefix (no boxed icon).
 */
function ExpandableStatusPill({
  status,
  open,
  onToggle,
}: {
  status?: string;
  open: boolean;
  onToggle: () => void;
}) {
  const s = normalizeStatus(status);
  const c = statusColors(s);
  const tooltip = STATUS_HELP[s] || STATUS_HELP.UNKNOWN;

  return (
    <HoverTip text={tooltip}>
      <button
        type="button"
        onClick={onToggle}
        aria-label={`Toggle lifecycle. Current status: ${s}`}
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          borderRadius: 999,
          padding: "3px 8px",
          fontSize: 10,
          fontWeight: 900,
          background: c.bg,
          border: `1px solid ${c.border}`,
          color: c.text,
          letterSpacing: 0.2,
          whiteSpace: "nowrap",
          cursor: "pointer",
        }}
      >
        <span style={{ display: "inline-flex", alignItems: "center" }}>
          {statusIconPrefix(s)}
          {s}
        </span>
        <span
          aria-hidden="true"
          style={{
            color: c.text,
            fontWeight: 900,
            lineHeight: 1,
            transform: open ? "translateY(-1px)" : "translateY(0)",
          }}
        >
          {open ? "▴" : "▾"}
        </span>
      </button>
    </HoverTip>
  );
}

function InfoIcon({
  onOpen,
  title,
  hoverText,
}: {
  onOpen: () => void;
  title?: string;
  hoverText?: string;
}) {
  // Use HoverTip everywhere; never use native title tooltips.
  const tipText = hoverText || title;

  const btn = (
    <button
      type="button"
      onClick={onOpen}
      aria-label={title || "More info"}
      style={{
        width: 18,
        height: 18,
        borderRadius: 999,
        border: "1px solid #cbd5e1",
        background: "#ffffff",
        color: "#475569",
        fontSize: 11,
        lineHeight: "16px",
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
        padding: 0,
        flex: "0 0 auto",
      }}
    >
      i
    </button>
  );

  return tipText ? <HoverTip text={tipText}>{btn}</HoverTip> : btn;
}

function CenterModal({
  content,
  onClose,
}: {
  content: ModalContent | null;
  onClose: () => void;
}) {
  const dialogRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!content) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    setTimeout(() => dialogRef.current?.focus(), 0);
    return () => window.removeEventListener("keydown", onKey);
  }, [content, onClose]);

  if (!content) return null;

  return (
    <div
      role="presentation"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 1000,
        background: "rgba(15,23,42,0.45)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 16,
      }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        tabIndex={-1}
        style={{
          width: "min(680px, 100%)",
          background: "#ffffff",
          borderRadius: 16,
          border: "1px solid #e2e8f0",
          boxShadow:
            "0 20px 50px rgba(15,23,42,0.25), 0 1px 3px rgba(15,23,42,0.12)",
          padding: 18,
          outline: "none",
        }}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 14, fontWeight: 800, color: "#0f172a" }}>
              {content.title}
            </div>
            <div
              style={{
                marginTop: 10,
                fontSize: 12,
                color: "#475569",
                lineHeight: 1.55,
                whiteSpace: "pre-wrap",
              }}
            >
              {content.body}
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            style={{
              border: "none",
              background: "transparent",
              color: "#64748b",
              cursor: "pointer",
              fontSize: 18,
              lineHeight: 1,
              padding: 0,
            }}
          >
            ×
          </button>
        </div>

        <div style={{ marginTop: 14, fontSize: 11, color: "#94a3b8" }}>
          Press <strong>Esc</strong> or click outside to close.
        </div>
      </div>
    </div>
  );
}

function ConfirmModal({
  content,
  onClose,
}: {
  content: ConfirmContent | null;
  onClose: () => void;
}) {
  const dialogRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!content) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    setTimeout(() => dialogRef.current?.focus(), 0);
    return () => window.removeEventListener("keydown", onKey);
  }, [content, onClose]);

  if (!content) return null;

  const confirmColor = content.danger ? "#b91c1c" : "#2563eb";

  return (
    <div
      role="presentation"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 1100,
        background: "rgba(15,23,42,0.45)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 16,
      }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        tabIndex={-1}
        style={{
          width: "min(620px, 100%)",
          background: "#ffffff",
          borderRadius: 16,
          border: "1px solid #e2e8f0",
          boxShadow:
            "0 20px 50px rgba(15,23,42,0.25), 0 1px 3px rgba(15,23,42,0.12)",
          padding: 18,
          outline: "none",
        }}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 14, fontWeight: 900, color: "#0f172a" }}>
              {content.title}
            </div>
            <div
              style={{
                marginTop: 10,
                fontSize: 12,
                color: "#475569",
                lineHeight: 1.55,
                whiteSpace: "pre-wrap",
              }}
            >
              {content.body}
            </div>

            <div
              style={{
                marginTop: 14,
                display: "flex",
                justifyContent: "flex-end",
                gap: 10,
              }}
            >
              <button
                type="button"
                onClick={onClose}
                aria-label={content.cancelLabel || "Cancel"}
                style={{
                  borderRadius: 999,
                  border: "1px solid #e2e8f0",
                  background: "#ffffff",
                  padding: "8px 12px",
                  fontSize: 12,
                  fontWeight: 800,
                  cursor: "pointer",
                }}
              >
                {content.cancelLabel || "Cancel"}
              </button>

              <button
                type="button"
                onClick={() => {
                  content.onConfirm();
                  onClose();
                }}
                aria-label={content.confirmLabel || "Confirm"}
                style={{
                  borderRadius: 999,
                  border: "none",
                  background: confirmColor,
                  color: "#ffffff",
                  padding: "8px 12px",
                  fontSize: 12,
                  fontWeight: 900,
                  cursor: "pointer",
                }}
              >
                {content.confirmLabel || "Confirm"}
              </button>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            style={{
              border: "none",
              background: "transparent",
              color: "#64748b",
              cursor: "pointer",
              fontSize: 18,
              lineHeight: 1,
              padding: 0,
            }}
          >
            ×
          </button>
        </div>

        <div style={{ marginTop: 10, fontSize: 11, color: "#94a3b8" }}>
          Press <strong>Esc</strong> or click outside to close.
        </div>
      </div>
    </div>
  );
}

function JobLifecycle({
  status,
  onOpenHelp,
}: {
  status?: string;
  onOpenHelp: () => void;
}) {
  const s = normalizeStatus(status);

  const steps = [
    { key: "SUBMITTED", label: "Submitted" },
    { key: "RUNNABLE", label: "Runnable" },
    { key: "STARTING", label: "Starting" },
    { key: "RUNNING", label: "Running" },
    { key: "DONE", label: "Done" },
  ] as const;

  const isTerminalSuccess = s === "SUCCEEDED";
  const isTerminalFail = s === "FAILED";
  const doneIcon = isTerminalSuccess ? "✓ " : isTerminalFail ? "⚠ " : "";

  const activeKey = isTerminalSuccess || isTerminalFail ? "DONE" : s;
  const activeIdx = Math.max(0, steps.findIndex((x) => x.key === activeKey));

  const pillStyle = (key: string, isActive: boolean, isPast: boolean) => {
    const colorKey =
      key === "DONE"
        ? isTerminalSuccess
          ? "SUCCEEDED"
          : isTerminalFail
          ? "FAILED"
          : "DONE"
        : key;
    const c = statusColors(colorKey);
    return {
      display: "inline-flex",
      alignItems: "center",
      fontSize: 10,
      padding: "4px 9px",
      borderRadius: 999,
      border: isActive ? `1px solid ${c.border}` : "1px solid #e2e8f0",
      background: isActive ? c.bg : isPast ? "#f8fafc" : "#ffffff",
      color: isActive ? c.text : "#475569",
      fontWeight: isActive ? 900 : 700,
      whiteSpace: "nowrap",
    } as React.CSSProperties;
  };

  return (
    <div style={{ marginTop: 10 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 10,
          flexWrap: "wrap",
          fontSize: 11,
          color: "#475569",
        }}
      >
        <div style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
          <strong style={{ color: "#0f172a" }}>Lifecycle:</strong>
          <InfoIcon
            onOpen={onOpenHelp}
            title="Status help"
            hoverText="Hover any status pill to see what it means."
          />
        </div>
      </div>

      <div
        style={{
          marginTop: 8,
          display: "flex",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 6,
        }}
      >
        {steps.map((step, i) => {
          const isActive = i === activeIdx;
          const isPast = i < activeIdx;

          const tooltip =
            step.key === "DONE"
              ? STATUS_HELP.DONE
              : STATUS_HELP[step.key] || STATUS_HELP.UNKNOWN;

          return (
            <div
              key={step.key}
              style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
            >
              <HoverTip text={tooltip}>
                <span style={pillStyle(step.key, isActive, isPast)}>
                  {step.key === "DONE" ? doneIcon : ""}
                  {step.label}
                </span>
              </HoverTip>

              {i < steps.length - 1 ? (
                <span style={{ color: "#94a3b8", fontWeight: 900 }}>→</span>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

type SortKey = "updated" | "created";
type SortDir = "desc" | "asc";

function triggerDownload(url: string) {
  const a = document.createElement("a");
  a.href = url;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

// ---------------------------
// Session persistence (v1)
// ---------------------------
const SESSION_KEY = "covid_cfr_console_state_v1";

type PersistedStateV1 = {
  v: 1;

  // table + client-only metadata
  recentJobs: JobStatus[];
  jobMetaById: Record<string, JobMeta>;
  jobInputsById: Record<string, SubmitPayload>;
  jobModelPackageById: Record<string, string>;

  // UI state worth restoring
  checkedIds: string[];
  selectedJobId: string | null;

  query: string;
  statusFilter: string;
  timeFilterMinutes: number;
  sortKey: SortKey;
  sortDir: SortDir;
  pageIndex: number;

  // small toggles
  lifecycleOpen: boolean;
  advancedOpen: boolean;

  // form state persistence
  jobNameDraft: string;
  jobDescriptionDraft: string;

  sampleChoice: string;
  useCustomSamples: boolean;
  customSamples: string;

  // persisted as string for backwards compatibility
  modelPackageChoice: string;

  outdirChoice: string;
  useCustomOutdir: boolean;
  customOutdir: string;

  // display-only; cannot restore real file attachment
  localFileName: string | null;
};

function getSessionStorage(): Storage | null {
  if (typeof window === "undefined") return null;
  // swap to localStorage if you want persistence across browser restarts
  return window.sessionStorage;
}

function readPersistedState(): PersistedStateV1 | null {
  const store = getSessionStorage();
  if (!store) return null;

  try {
    const raw = store.getItem(SESSION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as PersistedStateV1;
    if (!parsed || parsed.v !== 1) return null;
    return parsed;
  } catch {
    return null;
  }
}

function writePersistedState(state: PersistedStateV1) {
  const store = getSessionStorage();
  if (!store) return;

  try {
    store.setItem(SESSION_KEY, JSON.stringify(state));
  } catch {
    // best-effort only
  }
}

function App() {
  const [modal, setModal] = useState<ModalContent | null>(null);
  const [confirm, setConfirm] = useState<ConfirmContent | null>(null);
  const openModal = (title: string, body: string) => setModal({ title, body });

  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    const t = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, []);

  const [lifecycleOpen, setLifecycleOpen] = useState(false);

  // form state
  const [sampleChoice, setSampleChoice] = useState(SAMPLE_OPTIONS[0].value);
  const [useCustomSamples, setUseCustomSamples] = useState(false);
  const [customSamples, setCustomSamples] = useState("");

  // bundled model package (single dropdown)
  // ✅ IMPORTANT: explicitly type to union (prevents "lasso_v1" literal-only state)
  const [modelPackageChoice, setModelPackageChoice] =
    useState<ModelPackageChoice>(MODEL_PACKAGE_OPTIONS[0].value);

  const selectedModelPackage = useMemo(() => {
    return (
      MODEL_PACKAGES[modelPackageChoice] || MODEL_PACKAGES[MODEL_PACKAGE_OPTIONS[0].value]
    );
  }, [modelPackageChoice]);

  // keep payload variables (derived from package)
  const referenceChoice = selectedModelPackage.reference_fasta;
  const featureMatrixChoice = selectedModelPackage.train_feature_matrix;
  const modelChoice = selectedModelPackage.model;
  const scalerChoice = selectedModelPackage.scaler;

  const [outdirChoice, setOutdirChoice] = useState(OUTDIR_OPTIONS[0].value);
  const [useCustomOutdir, setUseCustomOutdir] = useState(false);
  const [customOutdir, setCustomOutdir] = useState("");

  const [localFileName, setLocalFileName] = useState<string | null>(null);

  // Local upload: browsers cannot restore <input type="file"> after reload.
  // Track whether a live File is currently attached in this runtime.
  const [hasLiveFile, setHasLiveFile] = useState(false);

  // advanced settings (fixed + collapsible)
  const [advancedOpen, setAdvancedOpen] = useState(false);

  // meta (client-side)
  const [jobName, setJobName] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [jobMetaById, setJobMetaById] = useState<Record<string, JobMeta>>({});
  const [jobInputsById, setJobInputsById] = useState<
    Record<string, SubmitPayload>
  >({});

  // store chosen package label for run summary
  const [jobModelPackageById, setJobModelPackageById] = useState<
    Record<string, string>
  >({});

  // submit/status
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitInfo, setSubmitInfo] = useState<string | null>(null);

  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [currentJob, setCurrentJob] = useState<JobStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);

  const MAX_JOBS = 50;
  const [recentJobs, setRecentJobs] = useState<JobStatus[]>([]);

  // multi-select
  const [checkedIds, setCheckedIds] = useState<Set<string>>(() => new Set());

  // search/filter
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("ALL");
  const [timeFilterMinutes, setTimeFilterMinutes] = useState<number>(-1);

  // sorting
  const [sortKey, setSortKey] = useState<SortKey>("updated");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  // pagination
  const PAGE_SIZE = 10;
  const [pageIndex, setPageIndex] = useState(0);

  // IMPORTANT: gate persistence until AFTER hydration has run
  const [isHydrated, setIsHydrated] = useState(false);

  const TIME_OPTIONS: Array<{ label: string; minutes: number }> = [
    { label: "Any time", minutes: -1 },
    { label: "Past 5 minutes", minutes: 5 },
    { label: "Past 15 minutes", minutes: 15 },
    { label: "Past 30 minutes", minutes: 30 },
    { label: "Past 1 hour", minutes: 60 },
    { label: "Past 6 hours", minutes: 360 },
    { label: "Past 24 hours", minutes: 1440 },
    { label: "Past 7 days", minutes: 10080 },
  ];

  // Demo-lock behavior: checkbox can show the custom field, but the dropdown controls the actual value.
  const effectiveSamples =
    useCustomSamples && !DEMO_LOCK_CUSTOM_S3 ? customSamples.trim() : sampleChoice;

  const effectiveOutdir =
    useCustomOutdir && !DEMO_LOCK_CUSTOM_S3 ? customOutdir.trim() : outdirChoice;

  // Keep the infra-logic/state for future product mode, but mirror the dropdown into custom values in demo-lock mode
  useEffect(() => {
    if (DEMO_LOCK_CUSTOM_S3 && useCustomSamples) setCustomSamples(sampleChoice);
  }, [useCustomSamples, sampleChoice]);

  useEffect(() => {
    if (DEMO_LOCK_CUSTOM_S3 && useCustomOutdir) setCustomOutdir(outdirChoice);
  }, [useCustomOutdir, outdirChoice]);

  // layout
  const width = useWindowWidth();
  const isNarrow = width < 980;

  const pageStyle: React.CSSProperties = {
    minHeight: "100vh",
    backgroundColor: "#f3f4f6",
    padding: isNarrow ? "20px 14px" : "28px 22px",
    boxSizing: "border-box",
  };

  const shellStyle: React.CSSProperties = {
    width: "100%",
    maxWidth: "1700px",
    margin: "0 auto",
  };

  const gridStyle: React.CSSProperties = {
    display: "grid",
    gridTemplateColumns: isNarrow ? "1fr" : "minmax(0, 1fr) minmax(0, 1fr)",
    gap: "18px",
    alignItems: "flex-start",
  };

  const cardStyle: React.CSSProperties = {
    backgroundColor: "#ffffff",
    borderRadius: "16px",
    boxShadow: "0 10px 25px rgba(15,23,42,0.04), 0 1px 3px rgba(15,23,42,0.06)",
    padding: "20px 24px",
    boxSizing: "border-box",
    overflow: "hidden",
  };

  const labelRowStyle: React.CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
  };

  const labelStyle: React.CSSProperties = {
    display: "block",
    fontSize: "13px",
    fontWeight: 600,
    color: "#0f172a",
    marginBottom: "4px",
  };

  const helperStyle: React.CSSProperties = {
    fontSize: "11px",
    color: "#64748b",
    marginBottom: "4px",
  };

  const inputStyle: React.CSSProperties = {
    width: "100%",
    borderRadius: "999px",
    border: "1px solid #e2e8f0",
    padding: "8px 14px",
    fontSize: "13px",
    boxSizing: "border-box",
  };

  const selectStyle: React.CSSProperties = {
    ...inputStyle,
    borderRadius: "999px",
  };

  const lockedFieldStyle: React.CSSProperties = {
    ...inputStyle,
    background: "#f8fafc",
    color: "#64748b",
    cursor: "not-allowed",
    opacity: 1,
  };

  const textAreaStyle: React.CSSProperties = {
    width: "100%",
    borderRadius: "14px",
    border: "1px solid #e2e8f0",
    padding: "10px 12px",
    fontSize: "13px",
    boxSizing: "border-box",
    resize: "vertical",
    minHeight: 70,
  };

  const baseButtonStyle: React.CSSProperties = {
    borderRadius: "999px",
    border: "none",
    backgroundColor: "#2563eb",
    color: "#ffffff",
    fontSize: "13px",
    fontWeight: 700,
    padding: "10px 18px",
    width: "100%",
  };

  // ---------------------------
  // Helpers that depend on state
  // ---------------------------
  function upsertRecentJobStable(job: JobStatus) {
    setRecentJobs((prev) => {
      const idx = prev.findIndex((j) => j.job_id === job.job_id);
      if (idx === -1) return [...prev, job].slice(-MAX_JOBS);
      const next = [...prev];
      next[idx] = { ...next[idx], ...job };
      return next;
    });
  }

  async function loadJob(jobId: string) {
    const id = jobId.trim();
    if (!id) return;
    setStatusError(null);
    try {
      const data = await fetchJobStatus(id);
      setCurrentJob(data);
      setSelectedJobId(id);
      upsertRecentJobStable(data);
    } catch (err: any) {
      setStatusError(err.message || "Failed to fetch job status");
    }
  }

  // ---------------------------
  // Hydrate from session on mount
  // ---------------------------
  useEffect(() => {
    const restored = readPersistedState();

    // after reload: no live file object exists
    setHasLiveFile(false);

    if (restored) {
      // Restore table + metadata
      setRecentJobs(Array.isArray(restored.recentJobs) ? restored.recentJobs : []);
      setJobMetaById(restored.jobMetaById || {});
      setJobInputsById(restored.jobInputsById || {});
      setJobModelPackageById(restored.jobModelPackageById || {});

      // Restore selection / highlight
      const restoredJobIds = new Set((restored.recentJobs || []).map((j) => j.job_id));
      const cleanedChecked = (restored.checkedIds || []).filter((id) => restoredJobIds.has(id));
      setCheckedIds(new Set(cleanedChecked));

      const restoredSelected =
        restored.selectedJobId && restoredJobIds.has(restored.selectedJobId)
          ? restored.selectedJobId
          : null;

      setSelectedJobId(restoredSelected);

      // If we have a selected job, refetch it so status is fresh
      if (restoredSelected) void loadJob(restoredSelected);

      // Restore filters/sort/paging
      setQuery(typeof restored.query === "string" ? restored.query : "");
      setStatusFilter(typeof restored.statusFilter === "string" ? restored.statusFilter : "ALL");
      setTimeFilterMinutes(
        typeof restored.timeFilterMinutes === "number" ? restored.timeFilterMinutes : -1
      );
      setSortKey(restored.sortKey === "created" ? "created" : "updated");
      setSortDir(restored.sortDir === "asc" ? "asc" : "desc");
      setPageIndex(typeof restored.pageIndex === "number" ? restored.pageIndex : 0);

      // Restore small toggles
      setLifecycleOpen(!!restored.lifecycleOpen);
      setAdvancedOpen(!!restored.advancedOpen);

      // Restore form drafts
      setJobName(typeof restored.jobNameDraft === "string" ? restored.jobNameDraft : "");
      setJobDescription(
        typeof restored.jobDescriptionDraft === "string" ? restored.jobDescriptionDraft : ""
      );

      // Restore submit form choices
      setSampleChoice(
        typeof restored.sampleChoice === "string" ? restored.sampleChoice : SAMPLE_OPTIONS[0].value
      );
      setUseCustomSamples(!!restored.useCustomSamples);
      setCustomSamples(typeof restored.customSamples === "string" ? restored.customSamples : "");

      // ✅ FIX: narrow persisted string into ModelPackageChoice
      const restoredPkg = restored.modelPackageChoice;
      setModelPackageChoice(
        isModelPackageChoice(restoredPkg) ? restoredPkg : MODEL_PACKAGE_OPTIONS[0].value
      );

      setOutdirChoice(
        typeof restored.outdirChoice === "string" ? restored.outdirChoice : OUTDIR_OPTIONS[0].value
      );
      setUseCustomOutdir(!!restored.useCustomOutdir);
      setCustomOutdir(typeof restored.customOutdir === "string" ? restored.customOutdir : "");

      // display-only
      setLocalFileName(
        typeof restored.localFileName === "string" || restored.localFileName === null
          ? restored.localFileName
          : null
      );
    }

    // Enable persistence only AFTER hydration has executed (prevents overwriting saved state)
    setIsHydrated(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------------------------
  // Persist to session (debounced)
  // ---------------------------
  const persistTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (!isHydrated) return;

    if (persistTimerRef.current) window.clearTimeout(persistTimerRef.current);

    persistTimerRef.current = window.setTimeout(() => {
      writePersistedState({
        v: 1,

        recentJobs,
        jobMetaById,
        jobInputsById,
        jobModelPackageById,

        checkedIds: Array.from(checkedIds),
        selectedJobId,

        query,
        statusFilter,
        timeFilterMinutes,
        sortKey,
        sortDir,
        pageIndex,

        lifecycleOpen,
        advancedOpen,

        jobNameDraft: jobName,
        jobDescriptionDraft: jobDescription,

        sampleChoice,
        useCustomSamples,
        customSamples,

        // ModelPackageChoice is assignable to string here
        modelPackageChoice,

        outdirChoice,
        useCustomOutdir,
        customOutdir,

        localFileName,
      });
    }, 250);

    return () => {
      if (persistTimerRef.current) window.clearTimeout(persistTimerRef.current);
    };
  }, [
    isHydrated,
    recentJobs,
    jobMetaById,
    jobInputsById,
    jobModelPackageById,
    checkedIds,
    selectedJobId,
    query,
    statusFilter,
    timeFilterMinutes,
    sortKey,
    sortDir,
    pageIndex,
    lifecycleOpen,
    advancedOpen,
    jobName,
    jobDescription,
    sampleChoice,
    useCustomSamples,
    customSamples,
    modelPackageChoice,
    outdirChoice,
    useCustomOutdir,
    customOutdir,
    localFileName,
  ]);

  const currentMeta = currentJob ? jobMetaById[currentJob.job_id] : undefined;
  const currentInputs = currentJob ? jobInputsById[currentJob.job_id] : undefined;

  const selectedOutdirLabel = useMemo(() => {
    const opt = OUTDIR_OPTIONS.find((o) => o.value === outdirChoice);
    return opt?.label || outdirChoice;
  }, [outdirChoice]);

  const canDownloadHighlighted = normalizeStatus(currentJob?.status) === "SUCCEEDED";
  const submitEnabled = jobName.trim().length > 0 && !isSubmitting;

  function openSelectedJobInfo() {
    if (!currentJob) return;

    const meta = jobMetaById[currentJob.job_id];
    const inputs = jobInputsById[currentJob.job_id];

    const pkgLabel =
      jobModelPackageById[currentJob.job_id] || MODEL_PACKAGES[MODEL_PACKAGE_OPTIONS[0].value].label;

    const nameLine = `Name: ${meta?.name || "— (not available)"}\n`;
    const idLine = `Job ID: ${currentJob.job_id}\n`;
    const descLine = `\nDescription:\n${meta?.description?.trim() || "—"}\n`;

    const advancedBlock =
      `\nAdvanced settings (fixed for demo):\n` +
      `• Min alignment identity: ${Math.round(ADVANCED_DEFAULTS.min_alignment_identity * 100)}%\n` +
      `• Chunk size (samples): ${ADVANCED_DEFAULTS.chunk_size_samples}\n` +
      `• Max branches: ${ADVANCED_DEFAULTS.max_branches}\n`;

    const inputsBlock = (() => {
      if (inputs) {
        return (
          `\nInputs used for submission (this browser):\n` +
          `• Samples: ${inputs.samples_uri}\n` +
          `• Model package: ${pkgLabel}\n` +
          `    - Reference: ${inputs.reference_fasta}\n` +
          `    - Train feature matrix: ${inputs.train_feature_matrix}\n` +
          `    - Model: ${inputs.model}\n` +
          `    - Scaler: ${inputs.scaler}\n` +
          `• Results folder: ${inputs.outdir}\n`
        );
      }

      const pkg = MODEL_PACKAGES[MODEL_PACKAGE_OPTIONS[0].value];

      return (
        `\nInputs used for submission:\n` +
        `— Not available (details are tracked only for jobs submitted in this browser.)\n` +
        `\nModel package (demo default):\n` +
        `• Model package: ${pkgLabel}\n` +
        `    - Reference: ${pkg.reference_fasta}\n` +
        `    - Train feature matrix: ${pkg.train_feature_matrix}\n` +
        `    - Model: ${pkg.model}\n` +
        `    - Scaler: ${pkg.scaler}\n`
      );
    })();

    openModal("Job details", nameLine + idLine + descLine + inputsBlock + advancedBlock);
  }

  function openLifecycleHelp() {
    openModal("Status help", "Hover any status pill to see what it means.\n\n✓ = succeeded\n⚠ = failed");
  }

  function openTableInfo() {
    openModal(
      "Table info",
      [
        "• Click a row to highlight a job.",
        "• Use checkboxes to download or remove multiple jobs.",
        "• Download selected: only SUCCEEDED jobs are downloadable.",
        "• Delete selected: removes completed jobs (SUCCEEDED/FAILED) from this browser list only.",
      ].join("\n")
    );
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitError(null);
    setSubmitInfo(null);

    if (!jobName.trim()) {
      setSubmitError("Please enter a job name.");
      return;
    }
    if (!effectiveSamples || !effectiveSamples.startsWith("s3://")) {
      setSubmitError("Input genomes must be an s3:// path.");
      return;
    }
    if (!effectiveOutdir || !effectiveOutdir.startsWith("s3://")) {
      setSubmitError("Results folder must be an s3:// path.");
      return;
    }

    const payload: SubmitPayload = {
      samples_uri: effectiveSamples,
      reference_fasta: referenceChoice,
      train_feature_matrix: featureMatrixChoice,
      model: modelChoice,
      scaler: scalerChoice,
      outdir: effectiveOutdir,
    };

    setIsSubmitting(true);
    try {
      const resp = await fetch(`${API_BASE}/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Failed to submit job");

      const jobId = data.job_id as string;
      const nowIso = new Date().toISOString();

      setJobMetaById((prev) => ({
        ...prev,
        [jobId]: { name: jobName.trim(), description: jobDescription.trim() },
      }));
      setJobInputsById((prev) => ({ ...prev, [jobId]: payload }));
      setJobModelPackageById((prev) => ({
        ...prev,
        [jobId]: selectedModelPackage.label,
      }));

      const initial: JobStatus = {
        job_id: jobId,
        status: data.status || "SUBMITTED",
        created_at: data.created_at || nowIso,
        updated_at: data.updated_at || nowIso,
        batch_job_id: data.batch_job_id,
      };

      setCurrentJob(initial);
      setSelectedJobId(jobId);
      upsertRecentJobStable(initial);

      setSubmitInfo(`Submitted job. Job ID: ${jobId}`);

      setQuery("");
      if (statusFilter !== "ALL" && statusFilter !== normalizeStatus(initial.status)) {
        setStatusFilter("ALL");
      }

      setJobName("");
      setJobDescription("");
      setLocalFileName(null);
      setHasLiveFile(false);

      void loadJob(jobId);
    } catch (err: any) {
      setSubmitError(err.message || "Failed to submit job");
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleDownload(job: JobStatus) {
    const url = job.download_url || `${API_BASE}/results/${job.job_id}/zip`;
    triggerDownload(url);
  }

  // --------------------------
  // Multi-download (no new API)
  // --------------------------
  const DOWNLOAD_CAP = 10;

  const recentById = useMemo(() => {
    const m = new Map<string, JobStatus>();
    for (const j of recentJobs) m.set(j.job_id, j);
    return m;
  }, [recentJobs]);

  const checkedList = useMemo(() => Array.from(checkedIds), [checkedIds]);

  const checkedJobs = useMemo(() => {
    const jobs: JobStatus[] = [];
    for (const id of checkedList) {
      const j = recentById.get(id);
      if (j) jobs.push(j);
    }
    return jobs;
  }, [checkedList, recentById]);

  const checkedSucceeded = useMemo(
    () => checkedJobs.filter((j) => normalizeStatus(j.status) === "SUCCEEDED"),
    [checkedJobs]
  );

  function downloadMany(jobs: JobStatus[], cap = DOWNLOAD_CAP) {
    const succeeded = jobs.filter((j) => normalizeStatus(j.status) === "SUCCEEDED");
    if (succeeded.length === 0) {
      openModal("No downloadable jobs selected", "Only jobs with status SUCCEEDED can be downloaded.");
      return;
    }

    const batch = succeeded.slice(0, cap);
    for (const job of batch) {
      const url = job.download_url || `${API_BASE}/results/${job.job_id}/zip`;
      triggerDownload(url);
    }

    if (succeeded.length > cap) {
      openModal(
        "Downloads started",
        `Opened ${cap} downloads.\n\n${succeeded.length - cap} more were not opened to avoid browser blocking.`
      );
    }
  }

  // --------------------------
  // Delete selected (client-side list removal)
  // Only completed jobs: SUCCEEDED or FAILED
  // --------------------------
  const COMPLETED_STATUSES = new Set(["SUCCEEDED", "FAILED"]);

  const checkedCompleted = useMemo(
    () => checkedJobs.filter((j) => COMPLETED_STATUSES.has(normalizeStatus(j.status))),
    [checkedJobs]
  );

  function deleteManyClientSide(ids: string[]) {
    if (ids.length === 0) return;

    setRecentJobs((prev) => prev.filter((j) => !ids.includes(j.job_id)));

    setCheckedIds((prev) => {
      const next = new Set(prev);
      for (const id of ids) next.delete(id);
      return next;
    });

    if (selectedJobId && ids.includes(selectedJobId)) {
      setSelectedJobId(null);
      setCurrentJob(null);
      setStatusError(null);
    }
  }

  // selection helpers
  function toggleChecked(id: string) {
    setCheckedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function setCheckedMany(ids: string[], checked: boolean) {
    setCheckedIds((prev) => {
      const next = new Set(prev);
      for (const id of ids) {
        if (checked) next.add(id);
        else next.delete(id);
      }
      return next;
    });
  }

  function clearChecked() {
    setCheckedIds(new Set());
  }

  // filtering/sorting
  const availableStatuses = useMemo(() => {
    const set = new Set<string>();
    for (const j of recentJobs) set.add(normalizeStatus(j.status));
    ["SUBMITTED", "RUNNABLE", "STARTING", "RUNNING", "SUCCEEDED", "FAILED"].forEach((s) => set.add(s));
    return Array.from(set).filter(Boolean).sort();
  }, [recentJobs]);

  function toggleSort(nextKey: SortKey) {
    if (sortKey === nextKey) {
      setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortKey(nextKey);
      setSortDir("desc");
    }
  }

  const filteredJobs = useMemo(() => {
    const q = query.trim().toLowerCase();
    const now = nowMs;

    const filtered = recentJobs.filter((job) => {
      const meta = jobMetaById[job.job_id];
      const haystack = `${job.job_id} ${(meta?.name || "")} ${meta?.description || ""}`.toLowerCase();

      if (q && !haystack.includes(q)) return false;

      if (statusFilter !== "ALL") {
        if (normalizeStatus(job.status) !== statusFilter) return false;
      }

      if (timeFilterMinutes !== -1) {
        const ms = updatedMs(job);
        if (ms === Number.NEGATIVE_INFINITY) return false;
        const ageMinutes = (now - ms) / 60000;
        if (ageMinutes > timeFilterMinutes) return false;
      }

      return true;
    });

    const keyFn = sortKey === "created" ? createdMs : updatedMs;
    filtered.sort((a, b) => {
      const diff = keyFn(b) - keyFn(a);
      return sortDir === "desc" ? diff : -diff;
    });

    return filtered;
  }, [recentJobs, jobMetaById, query, statusFilter, timeFilterMinutes, sortKey, sortDir, nowMs]);

  const totalPages = useMemo(
    () => Math.max(1, Math.ceil(filteredJobs.length / PAGE_SIZE)),
    [filteredJobs.length]
  );

  const safePageIndex = Math.min(pageIndex, totalPages - 1);

  useEffect(() => {
    if (pageIndex !== safePageIndex) setPageIndex(safePageIndex);
  }, [pageIndex, safePageIndex]);

  useEffect(() => {
    setPageIndex(0);
  }, [query, statusFilter, timeFilterMinutes, sortKey, sortDir]);

  const pagedJobs = useMemo(() => {
    const start = safePageIndex * PAGE_SIZE;
    return filteredJobs.slice(start, start + PAGE_SIZE);
  }, [filteredJobs, safePageIndex]);

  // If nothing highlighted but we have rows, highlight the first row in the table
  const firstRowId = pagedJobs[0]?.job_id || "";
  useEffect(() => {
    if (selectedJobId) return;
    if (!firstRowId) return;
    void loadJob(firstRowId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedJobId, firstRowId]);

  // keyboard paging
  useEffect(() => {
    const isTypingTarget = () => {
      const el = document.activeElement as HTMLElement | null;
      if (!el) return false;
      const tag = el.tagName.toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return true;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      if ((el as any).isContentEditable) return true;
      return false;
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (modal || confirm) return;
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      if (isTypingTarget()) return;

      if (e.key === "ArrowLeft") {
        if (safePageIndex > 0) {
          e.preventDefault();
          setPageIndex((p) => Math.max(0, p - 1));
        }
      } else if (e.key === "ArrowRight") {
        if (safePageIndex < totalPages - 1) {
          e.preventDefault();
          setPageIndex((p) => Math.min(totalPages - 1, p + 1));
        }
      } else if (e.key === "Home") {
        if (totalPages > 1) {
          e.preventDefault();
          setPageIndex(0);
        }
      } else if (e.key === "End") {
        if (totalPages > 1) {
          e.preventDefault();
          setPageIndex(totalPages - 1);
        }
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [modal, confirm, safePageIndex, totalPages]);

  const selectedSampleHint = useMemo(() => {
    const opt = SAMPLE_OPTIONS.find((o) => o.value === sampleChoice);
    return opt?.hint || "";
  }, [sampleChoice]);

  const updatedStamp = currentJob?.updated_at || currentJob?.created_at;
  const updatedRel = relativeTime(updatedStamp, nowMs);
  const createdStamp = currentJob?.created_at;
  const createdRel = relativeTime(createdStamp, nowMs);

  const sortArrow = (key: SortKey) => {
    const isActive = sortKey === key;
    const arrow = isActive ? (sortDir === "desc" ? "↓" : "↑") : "↕";
    return (
      <span
        style={{
          marginLeft: 6,
          color: isActive ? "#0f172a" : "#94a3b8",
          fontWeight: 800,
        }}
        aria-hidden="true"
      >
        {arrow}
      </span>
    );
  };

  const headerButtonStyle = (active: boolean): React.CSSProperties => ({
    border: "none",
    background: "transparent",
    padding: 0,
    margin: 0,
    cursor: "pointer",
    font: "inherit",
    color: active ? "#0f172a" : "#334155",
    fontWeight: active ? 800 : 700,
  });

  // Table sizing to avoid horizontal scroll
  const tableWrapperStyle: React.CSSProperties = {
    width: "100%",
    overflowX: "hidden",
  };

  const tableStyle: React.CSSProperties = {
    width: "100%",
    borderCollapse: "separate",
    borderSpacing: 0,
    fontSize: "11px",
    tableLayout: "fixed",
  };

  const thStyle: React.CSSProperties = {
    padding: "8px 8px",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  };

  const tdStyle: React.CSSProperties = {
    padding: "8px 8px",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
    verticalAlign: "middle",
  };

  // checkbox column should never ellipsis/clip
  const checkboxThStyle: React.CSSProperties = {
    ...thStyle,
    padding: "8px 6px",
    overflow: "visible",
    textOverflow: "clip",
    whiteSpace: "nowrap",
  };

  const checkboxTdStyle: React.CSSProperties = {
    padding: "8px 6px",
    verticalAlign: "middle",
    overflow: "visible",
    textOverflow: "clip",
    whiteSpace: "nowrap",
  };

  const cellBlock: React.CSSProperties = {
    display: "block",
    minWidth: 0,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  };

  // polling (hybrid)
  const PRIORITY_INTERVAL_MS = 5000;
  const RR_INTERVAL_MS = 12000;
  const PRIORITY_CAP = 10;
  const RR_CAP = 5;

  const priorityTimerRef = useRef<number | null>(null);
  const rrTimerRef = useRef<number | null>(null);

  const priorityInFlightRef = useRef(false);
  const rrInFlightRef = useRef(false);
  const rrCursorRef = useRef(0);

  const priorityIds = useMemo(() => {
    const ids = new Set<string>();
    if (currentJob?.job_id) ids.add(currentJob.job_id);
    for (const j of pagedJobs) ids.add(j.job_id);
    return Array.from(ids).slice(0, PRIORITY_CAP);
  }, [currentJob?.job_id, pagedJobs]);

  const priorityIdsKey = useMemo(() => priorityIds.join("|"), [priorityIds]);

  const rrPoolIds = useMemo(() => {
    const prioritySet = new Set(priorityIds);
    const pool = recentJobs
      .filter((j) => !prioritySet.has(j.job_id))
      .slice()
      .sort((a, b) => updatedMs(a) - updatedMs(b))
      .map((j) => j.job_id);
    return pool;
  }, [recentJobs, priorityIds]);

  const rrPoolKey = useMemo(() => rrPoolIds.join("|"), [rrPoolIds]);

  async function pollIds(ids: string[]) {
    if (ids.length === 0) return;
    const results = await Promise.allSettled(ids.map((id) => fetchJobStatus(id)));
    for (const r of results) {
      if (r.status !== "fulfilled") continue;
      const job = r.value;
      setCurrentJob((prev) => (prev?.job_id === job.job_id ? job : prev));
      upsertRecentJobStable(job);
    }
  }

  useEffect(() => {
    if (priorityTimerRef.current) {
      window.clearInterval(priorityTimerRef.current);
      priorityTimerRef.current = null;
    }
    if (priorityIds.length === 0) return;

    const tick = async () => {
      if (document.hidden) return;
      if (priorityInFlightRef.current) return;
      priorityInFlightRef.current = true;
      try {
        await pollIds(priorityIds);
      } finally {
        priorityInFlightRef.current = false;
      }
    };

    void tick();
    priorityTimerRef.current = window.setInterval(tick, PRIORITY_INTERVAL_MS);

    return () => {
      if (priorityTimerRef.current) {
        window.clearInterval(priorityTimerRef.current);
        priorityTimerRef.current = null;
      }
    };
  }, [priorityIdsKey]);

  useEffect(() => {
    if (rrTimerRef.current) {
      window.clearInterval(rrTimerRef.current);
      rrTimerRef.current = null;
    }
    if (rrPoolIds.length === 0) return;

    const tick = async () => {
      if (document.hidden) return;
      if (rrInFlightRef.current) return;

      rrInFlightRef.current = true;
      try {
        const n = rrPoolIds.length;
        if (n === 0) return;

        rrCursorRef.current = rrCursorRef.current % n;

        const ids: string[] = [];
        for (let i = 0; i < Math.min(RR_CAP, n); i++) {
          ids.push(rrPoolIds[(rrCursorRef.current + i) % n]);
        }

        rrCursorRef.current = (rrCursorRef.current + Math.min(RR_CAP, n)) % n;
        await pollIds(ids);
      } finally {
        rrInFlightRef.current = false;
      }
    };

    rrCursorRef.current = 0;

    void tick();
    rrTimerRef.current = window.setInterval(tick, RR_INTERVAL_MS);

    return () => {
      if (rrTimerRef.current) {
        window.clearInterval(rrTimerRef.current);
        rrTimerRef.current = null;
      }
    };
  }, [rrPoolKey]);

  // jump-to-selected UX
  const selectedIndexInFiltered = useMemo(() => {
    if (!selectedJobId) return -1;
    return filteredJobs.findIndex((j) => j.job_id === selectedJobId);
  }, [filteredJobs, selectedJobId]);

  const selectedPageIndex = useMemo(() => {
    if (selectedIndexInFiltered < 0) return null;
    return Math.floor(selectedIndexInFiltered / PAGE_SIZE);
  }, [selectedIndexInFiltered]);

  const selectedIsOnThisPage = useMemo(() => {
    if (selectedIndexInFiltered < 0) return false;
    const start = safePageIndex * PAGE_SIZE;
    return selectedIndexInFiltered >= start && selectedIndexInFiltered < start + PAGE_SIZE;
  }, [selectedIndexInFiltered, safePageIndex]);

  useEffect(() => {
    if (!selectedJobId) return;
    if (!selectedIsOnThisPage) return;
    const el = document.querySelector(`[data-rowid="${CSS.escape(selectedJobId)}"]`) as HTMLElement | null;
    if (!el) return;
    setTimeout(() => {
      el.scrollIntoView({ block: "center", behavior: "smooth" });
    }, 0);
  }, [selectedJobId, selectedIsOnThisPage, safePageIndex]);

  // header checkbox: select page
  const pageRowIds = useMemo(() => pagedJobs.map((j) => j.job_id), [pagedJobs]);

  const allOnPageChecked = useMemo(() => {
    return pageRowIds.length > 0 && pageRowIds.every((id) => checkedIds.has(id));
  }, [pageRowIds, checkedIds]);

  const someOnPageChecked = useMemo(() => {
    return pageRowIds.some((id) => checkedIds.has(id)) && !allOnPageChecked;
  }, [pageRowIds, checkedIds, allOnPageChecked]);

  const headerCheckboxRef = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    if (!headerCheckboxRef.current) return;
    headerCheckboxRef.current.indeterminate = someOnPageChecked;
  }, [someOnPageChecked]);

  // counts for button labels
  const selectedCount = checkedIds.size;
  const downloadableCount = checkedSucceeded.length;
  const deletableCount = checkedCompleted.length;

  // Clear filters disabled state
  const filtersApplied =
    query.trim().length > 0 ||
    statusFilter !== "ALL" ||
    timeFilterMinutes !== -1 ||
    sortKey !== "updated" ||
    sortDir !== "desc";

  // UI render helpers
  const arrowButtonStyle = (disabled: boolean): React.CSSProperties => ({
    width: 30,
    height: 26,
    borderRadius: 999,
    border: "1px solid #e2e8f0",
    background: "#ffffff",
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.4 : 1,
    fontWeight: 800,
    padding: 0,
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    lineHeight: 1,
  });

  const smallPillButton: React.CSSProperties = {
    fontSize: "11px",
    padding: "5px 10px",
    borderRadius: "999px",
    border: "1px solid #e2e8f0",
    background: "#ffffff",
    cursor: "pointer",
    fontWeight: 800,
  };

  const dangerPillButton: React.CSSProperties = {
    ...smallPillButton,
    borderColor: "#fecdd3",
    background: "#fff1f2",
    color: "#be123c",
    fontWeight: 900,
  };

  const renderFixedField = (value: string, hoverText?: string) => {
    const input = (
      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
        <input style={lockedFieldStyle} value={value} disabled readOnly />
      </div>
    );

    return hoverText ? (
      <HoverTip text={hoverText} block>
        {input}
      </HoverTip>
    ) : (
      input
    );
  };

  return (
    <div style={pageStyle}>
      <div style={shellStyle}>
        <header style={{ marginBottom: 18 }}>
          <h1
            style={{
              fontSize: "26px",
              fontWeight: 700,
              color: "#0f172a",
              margin: 0,
            }}
          >
            COVID-19 CFR Prediction Console (Demo)
          </h1>
          <p style={{ marginTop: 6, fontSize: "12px", color: "#64748b" }}>
            Run a CFR prediction job on SARS-CoV-2 genomes, track status live, and download results.
          </p>
        </header>

        <div style={gridStyle}>
          {/* LEFT CARD */}
          <section style={cardStyle}>
            <h2
              style={{
                fontSize: "15px",
                fontWeight: 700,
                color: "#0f172a",
                margin: 0,
                marginBottom: 12,
              }}
            >
              1. Submit a new job
            </h2>

            <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div>
                <div style={labelRowStyle}>
                  <label style={labelStyle}>
                    Job name <span style={{ color: "#b91c1c" }}>*</span>
                  </label>
                  <InfoIcon
                    onOpen={() =>
                      openModal("Job name", "Required. This is the human-friendly label shown to identify your run.")
                    }
                    title="Job name info"
                    hoverText="Human-friendly label for this run."
                  />
                </div>
                <input
                  style={inputStyle}
                  placeholder="e.g., Multi-file demo"
                  value={jobName}
                  maxLength={JOB_NAME_MAX}
                  onChange={(e) => setJobName(e.target.value.slice(0, JOB_NAME_MAX))}
                />
                <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 4 }}>
                  {jobName.length}/{JOB_NAME_MAX}
                </div>
              </div>

              <div>
                <div style={labelRowStyle}>
                  <label style={labelStyle}>Job description (optional)</label>
                  <InfoIcon
                    onOpen={() =>
                      openModal(
                        "Job description",
                        "Optional. A short note shown in this website for quick context (not sent to the backend)."
                      )
                    }
                    title="Job description info"
                    hoverText="Optional note shown in this UI."
                  />
                </div>
                <textarea
                  style={textAreaStyle}
                  placeholder="Example: “Demonstrates multi-file inputs and rejected records.”"
                  value={jobDescription}
                  maxLength={JOB_DESC_MAX}
                  onChange={(e) => setJobDescription(e.target.value.slice(0, JOB_DESC_MAX))}
                />
                <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 4 }}>
                  {jobDescription.length}/{JOB_DESC_MAX}
                </div>
              </div>

              <div>
                <div style={labelRowStyle}>
                  <label style={labelStyle}>Input genomes</label>
                  <InfoIcon
                    onOpen={() =>
                      openModal(
                        "Input genomes",
                        "SARS-CoV-2 genomes in FASTA format. The workflow validates input, runs predictions, and writes results to the configured results folder."
                      )
                    }
                    title="Input genomes info"
                    hoverText="Choose a demo or enter an s3:// FASTA path."
                  />
                </div>

                <select
                  style={selectStyle}
                  value={sampleChoice}
                  onChange={(e) => setSampleChoice(e.target.value)}
                  disabled={useCustomSamples && !DEMO_LOCK_CUSTOM_S3}
                >
                  {SAMPLE_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>

                {selectedSampleHint && <p style={{ ...helperStyle, marginTop: 6 }}>{selectedSampleHint}</p>}

                <div
                  style={{
                    marginTop: 6,
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    fontSize: "11px",
                    color: "#64748b",
                  }}
                >
                  <input
                    id="customSamples"
                    type="checkbox"
                    checked={useCustomSamples}
                    onChange={(e) => {
                      const checked = e.target.checked;
                      setUseCustomSamples(checked);
                      if (checked) {
                        setCustomSamples((prev) => (prev.trim() ? prev : sampleChoice));
                      }
                    }}
                    aria-label="Use custom S3 path for input genomes"
                  />

                  {DEMO_LOCK_CUSTOM_S3 ? (
                    <HoverTip text="Shown for realism. Custom S3 paths are locked in this demo build.">
                      <label htmlFor="customSamples" style={{ cursor: "help" }}>
                        Use custom S3 path
                      </label>
                    </HoverTip>
                  ) : (
                    <label htmlFor="customSamples">Use custom S3 path</label>
                  )}
                </div>

                {useCustomSamples && (
                  <div style={{ marginTop: 6 }}>
                    {DEMO_LOCK_CUSTOM_S3 ? (
                      <HoverTip
                        text="Demo mode: custom S3 paths are locked. The dropdown still controls the selected input."
                        block
                      >
                        <input style={lockedFieldStyle} value={sampleChoice} disabled readOnly />
                      </HoverTip>
                    ) : (
                      <TipWrap text="Custom S3 FASTA path (or glob)." block>
                        <input
                          style={inputStyle}
                          placeholder="s3://bucket/path/to/your_samples.fasta (or pattern)"
                          value={customSamples}
                          maxLength={URI_MAX}
                          onChange={(e) => setCustomSamples(e.target.value.slice(0, URI_MAX))}
                        />
                      </TipWrap>
                    )}
                  </div>
                )}
              </div>

              <div>
                <div style={labelRowStyle}>
                  <label style={labelStyle}>Upload local file (optional)</label>
                  <InfoIcon
                    onOpen={() =>
                      openModal(
                        "Local upload (demo note)",
                        "This control illustrates a realistic product UI. In this demo build, local upload is not connected to the backend."
                      )
                    }
                    title="Local upload info"
                    hoverText="UI-only demo (not connected to backend yet)."
                  />
                </div>

                <div style={{ marginTop: 6 }}>
                  <TipWrap text="UI-only demo: file is not uploaded to the backend yet." block>
                    <input
                      type="file"
                      onChange={(e) => {
                        const file = e.target.files?.[0] || null;
                        setLocalFileName(file ? file.name : null);
                        setHasLiveFile(!!file);
                      }}
                      style={{ fontSize: "11px" }}
                      aria-label="Select a local FASTA file (demo only)"
                    />
                  </TipWrap>
                </div>

                {localFileName && <p style={{ ...helperStyle, marginTop: 6 }}>Selected: {localFileName}</p>}

                {localFileName && !hasLiveFile && (
                  <p style={{ ...helperStyle, marginTop: 4, color: "#94a3b8" }}>
                    Note: after a reload, browsers do not keep the file attached. Please reselect the file to use it.
                  </p>
                )}
              </div>

              {/* Bundled model package */}
              <div>
                <div style={labelRowStyle}>
                  <label style={labelStyle}>Model package</label>
                  <InfoIcon
                    onOpen={() =>
                      openModal(
                        "Model package",
                        "A bundled, versioned set of artifacts that must stay together (reference + training feature matrix + model + scaler).\n\nThis demo keeps them fixed to avoid incompatible combinations."
                      )
                    }
                    title="Model package info"
                    hoverText="Bundled artifacts (fixed for demo)."
                  />
                </div>

                {MODEL_PACKAGE_OPTIONS.length <= 1 ? (
                  renderFixedField(selectedModelPackage.label, "Fixed for demo")
                ) : (
                  <select
                    style={selectStyle}
                    value={modelPackageChoice}
                    onChange={(e) => {
                      const v = e.target.value;
                      // ✅ FIX: narrow string -> ModelPackageChoice
                      if (isModelPackageChoice(v)) setModelPackageChoice(v);
                    }}
                  >
                    {MODEL_PACKAGE_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                )}

                <p style={{ ...helperStyle, marginTop: 6 }}>
                  Includes: Wuhan reference, fixed training features, Lasso model, scaler.
                </p>
              </div>

              <div>
                <div style={labelRowStyle}>
                  <label style={labelStyle}>Where results are written</label>
                  <InfoIcon
                    onOpen={() =>
                      openModal(
                        "Results location",
                        "Outputs are written to an S3 folder. When the job status becomes SUCCEEDED, a download becomes available."
                      )
                    }
                    title="Results location info"
                  />
                </div>

                {OUTDIR_OPTIONS.length <= 1 ? (
                  <HoverTip text="S3 folder where outputs are written." block>
                    <input
                      style={lockedFieldStyle}
                      value={selectedOutdirLabel}
                      disabled
                      readOnly
                      aria-label="Results folder (fixed)"
                    />
                  </HoverTip>
                ) : (
                  <HoverTip text="S3 folder where outputs are written." block>
                    <select
                      style={{
                        ...selectStyle,
                        ...(useCustomOutdir && !DEMO_LOCK_CUSTOM_S3
                          ? { cursor: "not-allowed", background: "#f8fafc", color: "#64748b" }
                          : null),
                      }}
                      value={outdirChoice}
                      onChange={(e) => {
                        const v = e.target.value;
                        setOutdirChoice(v);
                        if (useCustomOutdir && DEMO_LOCK_CUSTOM_S3) setCustomOutdir(v);
                      }}
                      disabled={useCustomOutdir && !DEMO_LOCK_CUSTOM_S3}
                      aria-label="Results folder preset"
                    >
                      {OUTDIR_OPTIONS.map((opt) => (
                        <option key={opt.value} value={opt.value}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  </HoverTip>
                )}

                <div
                  style={{
                    marginTop: 6,
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    fontSize: "11px",
                    color: "#64748b",
                  }}
                >
                  <input
                    id="customOutdir"
                    type="checkbox"
                    checked={useCustomOutdir}
                    onChange={(e) => {
                      const checked = e.target.checked;
                      setUseCustomOutdir(checked);
                      if (checked) {
                        setCustomOutdir((prev) => (prev.trim() ? prev : outdirChoice));
                      }
                    }}
                    aria-label="Use custom S3 folder for results"
                  />

                  {DEMO_LOCK_CUSTOM_S3 ? (
                    <HoverTip text="Shown for realism. Custom S3 folders are locked in this demo build.">
                      <label htmlFor="customOutdir" style={{ cursor: "help" }}>
                        Use custom S3 folder
                      </label>
                    </HoverTip>
                  ) : (
                    <label htmlFor="customOutdir">Use custom S3 folder</label>
                  )}
                </div>

                {useCustomOutdir && (
                  <div style={{ marginTop: 6 }}>
                    {DEMO_LOCK_CUSTOM_S3 ? (
                      <HoverTip
                        text="Demo mode: custom S3 folders are locked. The preset above controls the selected output."
                        block
                      >
                        <input
                          style={lockedFieldStyle}
                          value={customOutdir}
                          disabled
                          readOnly
                          aria-label="Custom results folder (locked)"
                        />
                      </HoverTip>
                    ) : (
                      <HoverTip text="Custom S3 folder where outputs will be written." block>
                        <input
                          style={inputStyle}
                          placeholder="s3://bucket/results/"
                          value={customOutdir}
                          maxLength={URI_MAX}
                          onChange={(e) => setCustomOutdir(e.target.value.slice(0, URI_MAX))}
                          aria-label="Custom results folder"
                        />
                      </HoverTip>
                    )}
                  </div>
                )}
              </div>

              {/* Advanced settings (fixed, collapsible) */}
              <div>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: 10,
                    flexWrap: "wrap",
                  }}
                >
                  <div style={labelRowStyle}>
                    <TipWrap text={advancedOpen ? "Hide advanced settings" : "Show advanced settings"}>
                      <button
                        type="button"
                        onClick={() => setAdvancedOpen((v) => !v)}
                        style={{
                          border: "none",
                          background: "transparent",
                          padding: 0,
                          margin: 0,
                          cursor: "pointer",
                          display: "inline-flex",
                          alignItems: "center",
                          gap: 6,
                          color: "#0f172a",
                        }}
                        aria-label={advancedOpen ? "Hide advanced settings" : "Show advanced settings"}
                      >
                        <span style={{ ...labelStyle, marginBottom: 0 }}>Advanced settings</span>
                        <span
                          aria-hidden="true"
                          style={{
                            fontSize: 12,
                            fontWeight: 900,
                            color: "#334155",
                            lineHeight: 1,
                            transform: advancedOpen ? "translateY(-1px)" : "translateY(0)",
                          }}
                        >
                          {advancedOpen ? "▴" : "▾"}
                        </span>
                      </button>
                    </TipWrap>

                    <InfoIcon
                      onOpen={() =>
                        openModal(
                          "Advanced settings",
                          "These parameters are fixed in the demo to keep the demo options predictable.\n\nThey are still shown so a reviewer can understand the run configuration."
                        )
                      }
                      title="Advanced settings info"
                      hoverText="Fixed for demo (shown for clarity)."
                    />
                  </div>

                  <TipWrap text={advancedOpen ? "Hide advanced settings" : "Show advanced settings"}>
                    <button
                      type="button"
                      onClick={() => setAdvancedOpen((v) => !v)}
                      style={{
                        fontSize: "11px",
                        padding: "5px 10px",
                        borderRadius: "999px",
                        border: "1px solid #e2e8f0",
                        background: advancedOpen ? "#f8fafc" : "#ffffff",
                        cursor: "pointer",
                        fontWeight: 900,
                        color: "#334155",
                      }}
                      aria-label={advancedOpen ? "Hide advanced settings" : "Show advanced settings"}
                    >
                      {advancedOpen ? "Hide" : "Show"}
                    </button>
                  </TipWrap>
                </div>

                {advancedOpen ? (
                  <div
                    style={{
                      marginTop: 10,
                      display: "grid",
                      gridTemplateColumns: isNarrow ? "1fr" : "repeat(3, minmax(0, 1fr))",
                      gap: 12,
                      alignItems: "start",
                    }}
                  >
                    <div>
                      <label style={labelStyle}>Min alignment identity</label>
                      <HoverTip text="Fixed for demo." block>
                        <input
                          style={lockedFieldStyle}
                          value={`${Math.round(ADVANCED_DEFAULTS.min_alignment_identity * 100)}%`}
                          disabled
                          readOnly
                        />
                      </HoverTip>
                      <div style={{ fontSize: 10, color: "#64748b", marginTop: 4, lineHeight: 1.35 }}>
                        Minimum match required for an alignment to pass.
                      </div>
                    </div>

                    <div>
                      <label style={labelStyle}>Chunk size (samples)</label>
                      <HoverTip text="Fixed for demo." block>
                        <input
                          style={lockedFieldStyle}
                          value={String(ADVANCED_DEFAULTS.chunk_size_samples)}
                          disabled
                          readOnly
                        />
                      </HoverTip>
                      <div style={{ fontSize: 10, color: "#64748b", marginTop: 4, lineHeight: 1.35 }}>
                        Number of samples per chunk emitted from the input and processed per branch.
                      </div>
                    </div>

                    <div>
                      <label style={labelStyle}>Max branches</label>
                      <HoverTip text="Fixed for demo." block>
                        <input
                          style={lockedFieldStyle}
                          value={String(ADVANCED_DEFAULTS.max_branches)}
                          disabled
                          readOnly
                        />
                      </HoverTip>
                      <div style={{ fontSize: 10, color: "#64748b", marginTop: 4, lineHeight: 1.35 }}>
                        Upper limit on how many branches may run in parallel.
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>

              <div style={{ marginTop: 6 }}>
                <TipWrap
                  text={
                    submitEnabled
                      ? "Submit a new job"
                      : !jobName.trim()
                      ? "Enter a job name to submit."
                      : "Submitting…"
                  }
                  block
                >
                  <button
                    type="submit"
                    style={{
                      ...baseButtonStyle,
                      cursor: submitEnabled ? "pointer" : "not-allowed",
                      opacity: submitEnabled ? 1 : 0.5,
                    }}
                    disabled={!submitEnabled}
                    aria-label="Submit job"
                  >
                    {isSubmitting ? "Submitting…" : "Submit job"}
                  </button>
                </TipWrap>

                {submitError && <p style={{ marginTop: 6, fontSize: "11px", color: "#b91c1c" }}>{submitError}</p>}
                {submitInfo && <p style={{ marginTop: 6, fontSize: "11px", color: "#166534" }}>{submitInfo}</p>}
              </div>
            </form>

            <p style={{ marginTop: 10, fontSize: "10px", color: "#94a3b8" }}>
              API base: <code>{API_BASE}</code>
            </p>
          </section>

          {/* RIGHT CARD */}
          <section style={cardStyle}>
            <h2
              style={{
                fontSize: "15px",
                fontWeight: 700,
                color: "#0f172a",
                margin: 0,
                marginBottom: 12,
              }}
            >
              2. Search jobs & download results
            </h2>

            {/* Controls row */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: isNarrow
                  ? "1fr"
                  : "minmax(360px, 1.2fr) minmax(180px, 0.8fr) minmax(180px, 0.8fr)",
                gap: 10,
                marginBottom: 10,
              }}
            >
              <div>
                <label style={labelStyle}>Search</label>
                <input
                  style={inputStyle}
                  placeholder="Search by job ID, job name, or description…"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
                  <TipWrap text={filtersApplied ? "Clear search, status, time, and sort" : "No filters are applied"}>
                    <button
                      type="button"
                      style={{
                        fontSize: "11px",
                        padding: "5px 10px",
                        borderRadius: "999px",
                        border: "1px solid #e2e8f0",
                        background: "#ffffff",
                        cursor: filtersApplied ? "pointer" : "not-allowed",
                        fontWeight: 800,
                        opacity: filtersApplied ? 1 : 0.5,
                      }}
                      onClick={() => {
                        if (!filtersApplied) return;
                        setQuery("");
                        setStatusFilter("ALL");
                        setTimeFilterMinutes(-1);
                        setSortKey("updated");
                        setSortDir("desc");
                      }}
                      disabled={!filtersApplied}
                      aria-label="Clear filters"
                    >
                      Clear filters
                    </button>
                  </TipWrap>

                  <span style={{ fontSize: "11px", color: "#64748b", alignSelf: "center" }}>
                    {filteredJobs.length} / {recentJobs.length} shown
                  </span>
                </div>
              </div>

              <div>
                <label style={labelStyle}>Status</label>
                <select
                  style={selectStyle}
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  aria-label="Status filter"
                >
                  <option value="ALL">All</option>
                  {availableStatuses.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label style={labelStyle}>Updated</label>
                <select
                  style={selectStyle}
                  value={String(timeFilterMinutes)}
                  onChange={(e) => setTimeFilterMinutes(Number(e.target.value))}
                  aria-label="Updated time filter"
                >
                  {TIME_OPTIONS.map((opt) => (
                    <option key={opt.minutes} value={String(opt.minutes)}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Selected job panel */}
            {currentJob && (
              <div style={{ marginTop: 6, marginBottom: 14, fontSize: "12px", color: "#0f172a" }}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    justifyContent: "space-between",
                    gap: 10,
                  }}
                >
                  <div style={{ minWidth: 0 }}>
                    <div
                      style={{
                        fontSize: 14,
                        fontWeight: 800,
                        color: "#0f172a",
                        lineHeight: 1.2,
                        wordBreak: "break-word",
                      }}
                    >
                      {currentMeta?.name || "Highlighted job"}
                    </div>

                    <div
                      style={{
                        marginTop: 4,
                        fontSize: 11,
                        color: "#64748b",
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        flexWrap: "wrap",
                      }}
                    >
                      <span>Job ID:</span>
                      <TipWrap text={currentJob.job_id} maxWidth={520}>
                        <code
                          style={{
                            color: "#0f172a",
                            maxWidth: "100%",
                            whiteSpace: "normal",
                            wordBreak: "break-all",
                            overflowWrap: "anywhere",
                          }}
                        >
                          {currentJob.job_id}
                        </code>
                      </TipWrap>
                    </div>
                  </div>

                  <InfoIcon
                    onOpen={openSelectedJobInfo}
                    title="Job details"
                    hoverText="View inputs + metadata for highlighted job."
                  />
                </div>

                <div style={{ marginTop: 10 }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: "#0f172a" }}>Description</div>
                  <div
                    style={{
                      marginTop: 4,
                      fontSize: 11,
                      color: currentMeta?.description?.trim() ? "#475569" : "#94a3b8",
                      minHeight: 16,
                      wordBreak: "break-word",
                    }}
                  >
                    {currentMeta?.description?.trim() || "—"}
                  </div>
                </div>

                <div
                  style={{
                    margin: "10px 0 0",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: 10,
                    flexWrap: "wrap",
                  }}
                >
                  <div style={{ display: "inline-flex", alignItems: "center", gap: 8, flexWrap: "wrap", minWidth: 0 }}>
                    <strong>Status:</strong>
                    <ExpandableStatusPill
                      status={currentJob.status}
                      open={lifecycleOpen}
                      onToggle={() => setLifecycleOpen((v) => !v)}
                    />
                  </div>

                  <TipWrap text={lifecycleOpen ? "Hide lifecycle" : "Show lifecycle"}>
                    <button
                      type="button"
                      onClick={() => setLifecycleOpen((v) => !v)}
                      style={{
                        ...smallPillButton,
                        fontWeight: 900,
                        background: lifecycleOpen ? "#f8fafc" : "#ffffff",
                        color: "#334155",
                        borderColor: "#e2e8f0",
                      }}
                      aria-label={lifecycleOpen ? "Hide lifecycle" : "Show lifecycle"}
                    >
                      {lifecycleOpen ? "Hide lifecycle" : "Show lifecycle"}
                    </button>
                  </TipWrap>
                </div>

                {lifecycleOpen ? <JobLifecycle status={currentJob.status} onOpenHelp={openLifecycleHelp} /> : null}

                {/* ONE timestamp block (no duplicates) */}
                <div
                  style={{
                    marginTop: 10,
                    display: "flex",
                    flexDirection: "column",
                    gap: 2,
                    fontSize: 11,
                    color: "#0f172a",
                  }}
                >
                  <div>
                    <strong>Submitted:</strong> {formatTime(createdStamp)}
                    {createdRel ? <span style={{ color: "#64748b" }}> ({createdRel})</span> : null}
                  </div>

                  <div>
                    <strong>Last updated:</strong> {formatTime(updatedStamp)}
                    {updatedRel ? <span style={{ color: "#64748b" }}> ({updatedRel})</span> : null}
                  </div>
                </div>

                <TipWrap
                  text={
                    canDownloadHighlighted
                      ? "Downloads results for the highlighted row."
                      : "Highlight a SUCCEEDED job to enable download."
                  }
                  block
                >
                  <button
                    type="button"
                    style={{
                      ...baseButtonStyle,
                      marginTop: 10,
                      opacity: canDownloadHighlighted ? 1 : 0.5,
                      cursor: canDownloadHighlighted ? "pointer" : "not-allowed",
                    }}
                    onClick={() => currentJob && handleDownload(currentJob)}
                    disabled={!canDownloadHighlighted}
                    aria-label="Download highlighted job"
                  >
                    Download highlighted job
                  </button>
                </TipWrap>

                {!canDownloadHighlighted && (
                  <p style={{ marginTop: 4, fontSize: "10px", color: "#94a3b8" }}>
                    Download becomes available when status is <strong>SUCCEEDED</strong>.
                  </p>
                )}

                {statusError && <p style={{ marginTop: 8, fontSize: "11px", color: "#b91c1c" }}>{statusError}</p>}

                {!currentInputs && (
                  <p style={{ marginTop: 6, fontSize: "10px", color: "#94a3b8" }}>
                    Tip: job details include inputs only for jobs submitted in this browser session.
                  </p>
                )}
              </div>
            )}

            {/* Table */}
            {recentJobs.length === 0 ? (
              <p style={{ fontSize: "11px", color: "#64748b" }}>Submit a job to see it listed here.</p>
            ) : filteredJobs.length === 0 ? (
              <p style={{ fontSize: "11px", color: "#64748b" }}>No jobs match the current filters.</p>
            ) : (
              <>
                {/* Selected row off-page banner */}
                {selectedJobId && selectedIndexInFiltered >= 0 && !selectedIsOnThisPage && selectedPageIndex !== null && (
                  <div
                    style={{
                      marginBottom: 10,
                      padding: "8px 10px",
                      borderRadius: 12,
                      border: "1px solid #e2e8f0",
                      background: "#f8fafc",
                      fontSize: 11,
                      color: "#334155",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: 10,
                    }}
                  >
                    <div style={{ minWidth: 0 }}>
                      Highlighted job is not on this page. It is currently on <strong>page {selectedPageIndex + 1}</strong>.
                    </div>

                    <TipWrap text="Jump to the page containing the highlighted job">
                      <button
                        type="button"
                        onClick={() => setPageIndex(selectedPageIndex)}
                        style={{
                          ...smallPillButton,
                          borderColor: "#bfdbfe",
                          background: "#eff6ff",
                          color: "#1d4ed8",
                        }}
                        aria-label={`Jump to page ${selectedPageIndex + 1}`}
                      >
                        Jump to page {selectedPageIndex + 1}
                      </button>
                    </TipWrap>
                  </div>
                )}

                {selectedJobId && selectedIndexInFiltered < 0 && (
                  <div
                    style={{
                      marginBottom: 10,
                      padding: "8px 10px",
                      borderRadius: 12,
                      border: "1px solid #e2e8f0",
                      background: "#f8fafc",
                      fontSize: 11,
                      color: "#64748b",
                    }}
                  >
                    A job is highlighted, but it is not in the current filtered results. (It may be hidden by search/status/time filters.)
                  </div>
                )}

                {/* Pagination + selection toolbar */}
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: 10,
                    marginBottom: 8,
                    fontSize: 11,
                    color: "#475569",
                    flexWrap: "wrap",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <TipWrap text="Previous page (←)">
                      <button
                        type="button"
                        onClick={() => setPageIndex((p) => Math.max(0, p - 1))}
                        disabled={safePageIndex === 0}
                        aria-label="Previous page"
                        style={arrowButtonStyle(safePageIndex === 0)}
                      >
                        ←
                      </button>
                    </TipWrap>

                    <TipWrap text="Next page (→)">
                      <button
                        type="button"
                        onClick={() => setPageIndex((p) => Math.min(totalPages - 1, p + 1))}
                        disabled={safePageIndex >= totalPages - 1}
                        aria-label="Next page"
                        style={arrowButtonStyle(safePageIndex >= totalPages - 1)}
                      >
                        →
                      </button>
                    </TipWrap>

                    <span>
                      Page <strong>{safePageIndex + 1}</strong> of <strong>{totalPages}</strong>
                    </span>

                    <span style={{ color: "#94a3b8" }}>•</span>
                    <span style={{ color: "#64748b" }}>Use ← / → keys</span>
                  </div>

                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <TipWrap
                      text={
                        selectedCount === 0
                          ? "Select rows to download."
                          : downloadableCount === 0
                          ? "No selected rows are SUCCEEDED yet."
                          : "Download all selected SUCCEEDED jobs (your browser may ask to allow multiple downloads)."
                      }
                    >
                      <button
                        type="button"
                        onClick={() => downloadMany(checkedJobs)}
                        disabled={downloadableCount === 0}
                        style={{
                          ...smallPillButton,
                          borderColor: downloadableCount > 0 ? "#a7f3d0" : "#e2e8f0",
                          background: downloadableCount > 0 ? "#ecfdf5" : "#ffffff",
                          color: downloadableCount > 0 ? "#047857" : "#94a3b8",
                          cursor: downloadableCount > 0 ? "pointer" : "not-allowed",
                          fontWeight: 900,
                        }}
                        aria-label="Download selected"
                      >
                        Download selected ({downloadableCount})
                      </button>
                    </TipWrap>

                    <TipWrap text={selectedCount === 0 ? "No selected rows." : "Clear selected rows."}>
                      <button
                        type="button"
                        onClick={() => clearChecked()}
                        disabled={selectedCount === 0}
                        style={{
                          ...smallPillButton,
                          opacity: selectedCount === 0 ? 0.5 : 1,
                          cursor: selectedCount === 0 ? "not-allowed" : "pointer",
                        }}
                        aria-label="Clear selection"
                      >
                        Clear selection ({selectedCount})
                      </button>
                    </TipWrap>

                    <TipWrap
                      text={
                        selectedCount === 0
                          ? "Select rows first."
                          : deletableCount === 0
                          ? "Only completed jobs (SUCCEEDED or FAILED) can be removed from this table."
                          : "Delete completed selected jobs from this table (client-side only)."
                      }
                    >
                      <button
                        type="button"
                        onClick={() => {
                          if (selectedCount === 0) return;

                          if (deletableCount === 0) {
                            openModal("Nothing deletable selected", "Delete selected only applies to completed jobs (SUCCEEDED or FAILED).");
                            return;
                          }

                          setConfirm({
                            title: "Delete selected jobs?",
                            body:
                              `This will remove ${deletableCount} completed job(s) from the table in this browser.\n\n` +
                              `This does NOT cancel AWS Batch jobs or delete S3 data.\n\n` +
                              `Proceed?`,
                            danger: true,
                            confirmLabel: "Delete",
                            cancelLabel: "Cancel",
                            onConfirm: () => {
                              deleteManyClientSide(checkedCompleted.map((j) => j.job_id));
                            },
                          });
                        }}
                        disabled={deletableCount === 0}
                        style={{
                          ...dangerPillButton,
                          opacity: deletableCount === 0 ? 0.5 : 1,
                          cursor: deletableCount === 0 ? "not-allowed" : "pointer",
                        }}
                        aria-label="Delete selected"
                      >
                        Delete selected ({deletableCount})
                      </button>
                    </TipWrap>
                  </div>

                  <div style={{ color: "#64748b" }}>
                    {filteredJobs.length === 0
                      ? "0 results"
                      : (() => {
                          const start = safePageIndex * PAGE_SIZE + 1;
                          const end = Math.min((safePageIndex + 1) * PAGE_SIZE, filteredJobs.length);
                          return `Showing ${start}-${end} of ${filteredJobs.length}`;
                        })()}
                  </div>
                </div>

                <div style={tableWrapperStyle}>
                  <table style={tableStyle}>
                    <colgroup>
                      <col style={{ width: 44 }} />
                      <col style={{ width: "20%" }} />
                      <col style={{ width: "16%" }} />
                      <col style={{ width: "18%" }} />
                      <col style={{ width: "21%" }} />
                      <col style={{ width: "21%" }} />
                    </colgroup>

                    <thead>
                      <tr style={{ textAlign: "left", backgroundColor: "#f8fafc" }}>
                        <th style={checkboxThStyle}>
                          <TipWrap text="Select all rows on this page">
                            <input
                              ref={headerCheckboxRef}
                              type="checkbox"
                              checked={allOnPageChecked}
                              onChange={(e) => {
                                setCheckedMany(pageRowIds, e.target.checked);
                              }}
                              aria-label="Select all rows on this page"
                            />
                          </TipWrap>
                        </th>

                        <th style={thStyle}>Name</th>
                        <th style={{ ...thStyle, paddingRight: 4 }}>Job ID</th>
                        <th style={{ ...thStyle, paddingLeft: 4, paddingRight: 14 }}>Status</th>

                        <th style={{ ...thStyle, paddingLeft: 12 }}>
                          <TipWrap text="Sort by submission time">
                            <button
                              type="button"
                              style={headerButtonStyle(sortKey === "created")}
                              onClick={() => toggleSort("created")}
                              aria-label="Sort by submission time"
                            >
                              Submitted {sortArrow("created")}
                            </button>
                          </TipWrap>
                        </th>

                        <th style={thStyle}>
                          <TipWrap text="Sort by last updated time">
                            <button
                              type="button"
                              style={headerButtonStyle(sortKey === "updated")}
                              onClick={() => toggleSort("updated")}
                              aria-label="Sort by last updated time"
                            >
                              Updated {sortArrow("updated")}
                            </button>
                          </TipWrap>
                        </th>
                      </tr>
                    </thead>

                    <tbody>
                      {pagedJobs.map((job) => {
                        const meta = jobMetaById[job.job_id];
                        const isSelected = selectedJobId === job.job_id;
                        const isChecked = checkedIds.has(job.job_id);

                        const cStamp = job.created_at;
                        const uStamp = job.updated_at || job.created_at;

                        const descTip = meta?.description?.trim() ? meta.description.trim() : undefined;

                        return (
                          <tr
                            key={job.job_id}
                            data-rowid={job.job_id}
                            onClick={() => void loadJob(job.job_id)}
                            style={{
                              cursor: "pointer",
                              backgroundColor: isSelected ? "#eff6ff" : "transparent",
                              borderLeft: isSelected ? "3px solid #2563eb" : "3px solid transparent",
                            }}
                          >
                            <td style={checkboxTdStyle} onClick={(e) => e.stopPropagation()}>
                              <TipWrap text="Select for multi-download">
                                <input
                                  type="checkbox"
                                  checked={isChecked}
                                  onChange={() => toggleChecked(job.job_id)}
                                  aria-label={`Select job ${job.job_id}`}
                                />
                              </TipWrap>
                            </td>

                            <td style={{ ...tdStyle, color: meta?.name ? "#0f172a" : "#94a3b8" }}>
                              <TipWrap text={descTip || meta?.name || undefined} maxWidth={520} block>
                                <span style={cellBlock}>{meta?.name || "—"}</span>
                              </TipWrap>
                            </td>

                            <td style={{ ...tdStyle, paddingRight: 4 }}>
                              <TipWrap text={job.job_id} maxWidth={520}>
                                <span style={cellBlock}>
                                  <code
                                    style={{
                                      display: "inline-block",
                                      maxWidth: "100%",
                                      overflow: "hidden",
                                      textOverflow: "ellipsis",
                                      whiteSpace: "nowrap",
                                      color: "#0f172a",
                                    }}
                                  >
                                    {truncateId(job.job_id)}
                                  </code>
                                </span>
                              </TipWrap>
                            </td>

                            <td style={{ ...tdStyle, paddingLeft: 4, paddingRight: 14 }}>
                              <StatusPill status={job.status} />
                            </td>

                            <td style={{ ...tdStyle, paddingLeft: 12 }}>
                              <TipWrap text={cStamp || undefined} maxWidth={520}>
                                <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                                  <span style={cellBlock}>{formatTimeCompact(cStamp)}</span>
                                  <span style={{ fontSize: 10, color: "#64748b" }}>
                                    {cStamp ? relativeTime(cStamp, nowMs) : ""}
                                  </span>
                                </div>
                              </TipWrap>
                            </td>

                            <td style={tdStyle}>
                              <TipWrap text={uStamp || undefined} maxWidth={520}>
                                <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                                  <span style={cellBlock}>{formatTimeCompact(uStamp)}</span>
                                  <span style={{ fontSize: 10, color: "#64748b" }}>
                                    {uStamp ? relativeTime(uStamp, nowMs) : ""}
                                  </span>
                                </div>
                              </TipWrap>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div
                  style={{
                    marginTop: 8,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "flex-end",
                    gap: 8,
                    fontSize: 11,
                    color: "#64748b",
                  }}
                >
                  <span>Table info</span>
                  <InfoIcon onOpen={openTableInfo} title="Table info" hoverText="How selection, download, and delete work." />
                </div>
              </>
            )}
          </section>
        </div>
      </div>

      <CenterModal content={modal} onClose={() => setModal(null)} />
      <ConfirmModal content={confirm} onClose={() => setConfirm(null)} />
    </div>
  );
}

export default App;
