import http from "k6/http";
import { check, sleep } from "k6";
import { Rate, Trend } from "k6/metrics";

const API_BASE = __ENV.API_BASE; // e.g. https://...execute-api.../dev
const API_KEY = __ENV.API_KEY;   // your x-api-key VALUE
const OUTDIR = __ENV.OUTDIR || "s3://ach-covid-lasso-us-east-2/results";

// Explicit metrics (so throttling doesn't look like "random failures")
export const throttled = new Rate("throttled");           // 429
export const server_errors = new Rate("server_errors");   // 5xx
export const other_4xx = new Rate("other_4xx");           // 4xx excluding 429
export const success_rate = new Rate("success_rate");     // 2xx only
export const success_req_duration = new Trend("success_req_duration"); // ms, 2xx only

export const options = {
  scenarios: {
    steady: {
      executor: "constant-arrival-rate",
      rate: Number(__ENV.RATE || 2), // requests/sec
      timeUnit: "1s",
      duration: __ENV.DURATION || "2m",
      preAllocatedVUs: Number(__ENV.VUS || 10),
      maxVUs: Number(__ENV.MAX_VUS || 50),
    },
  },
  thresholds: {
    // Reliability: true server faults should be near-zero
    server_errors: ["rate<0.001"],      // <0.1% 5xx

    // Optional: non-throttle 4xx should be very low
    other_4xx: ["rate<0.01"],

    // Tail latency for *successful* requests only
    success_req_duration: ["p(95)<800"],

    // Throttling is expected when you intentionally overdrive
    // Tune this to your desired “how much 429 is acceptable under load?”
    throttled: ["rate<0.85"],
  },
};

function payload() {
  return JSON.stringify({
    phase: "init",
    reference_fasta:
      "s3://ach-covid-lasso-us-east-2/inputs/reference/NC_045512.2_sequence.fasta",
    train_feature_matrix:
      "s3://ach-covid-lasso-us-east-2/inputs/lasso/feature_matrix_train.csv",
    model: "s3://ach-covid-lasso-us-east-2/inputs/model/lasso_model.joblib",
    scaler: "s3://ach-covid-lasso-us-east-2/inputs/model/scaler.joblib",
    outdir: OUTDIR,
    files: [{ filename: "tiny.fasta", content_type: "text/plain", size_bytes: 50 }],
  });
}

export default function () {
  if (!API_BASE || !API_KEY) throw new Error("Set API_BASE and API_KEY env vars");

  const res = http.post(`${API_BASE}/submit`, payload(), {
    headers: { "content-type": "application/json", "x-api-key": API_KEY },
    timeout: "30s",
  });

  const is2xx = res.status >= 200 && res.status < 300;

  // classify outcomes
  throttled.add(res.status === 429);
  server_errors.add(res.status >= 500);
  other_4xx.add(res.status >= 400 && res.status < 500 && res.status !== 429);
  success_rate.add(is2xx);

  // success-only latency
  if (is2xx) {
    success_req_duration.add(res.timings.duration);
  }

  check(res, {
    "status is 2xx/429": (r) => (r.status >= 200 && r.status < 300) || r.status === 429,
    "has job_id when 2xx": (r) => {
      if (!(r.status >= 200 && r.status < 300)) return true;
      try {
        return !!JSON.parse(r.body).job_id;
      } catch {
        return false;
      }
    },
  });

  sleep(0.1);
}
