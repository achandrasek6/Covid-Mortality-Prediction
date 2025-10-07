# root/app/main.py

'''
# --- make a payload that should push prediction UP
cat > /tmp/push_up.json <<'JSON'
{
  "sample_id": "push-up",
  "features": {
    "S_3527": 1,
    "S_645": 1,
    "ORF1ab_2428": 1,
    "S_1451": 1,
    "S_571": 1,
    "S_53": 1,
    "ORF1ab_469": 1,
    "ORF1ab_11809": 1
  }
}
JSON

# --- make a payload that should push prediction DOWN
cat > /tmp/push_down.json <<'JSON'
{
  "sample_id": "push-down",
  "features": {
    "S_1434": 1,
    "S_426": 1
  }
}
JSON

# TEST CALL:
curl -sS -X POST http://127.0.0.1:8000/predict   -H 'Content-Type: application/json'   --data-binary @/tmp/push_up.json | jq .
curl -sS -X POST http://127.0.0.1:8000/predict   -H 'Content-Type: application/json'   --data-binary @/tmp/push_down.json | jq .

Feature-space pinning demo
--------------------------
# 1) Find the observed SHA:
curl -s http://127.0.0.1:8000/version
# → {"feature_file_sha":"80d0645637d5", ...}

# 2) Enforce it (startup will fail if header bytes change):
export EXPECTED_FEATURE_SHA=80d0645637d5
export FEATURE_SHA_ENFORCE=strict
uvicorn app.main:app --reload --port 8000

# 3) To allow boot but warn (not recommended for prod):
export FEATURE_SHA_ENFORCE=warn

Feature List Endpoint
-----------------------------
# JSON
curl -s http://127.0.0.1:8000/features | jq

# CSV (header-only, with download filename)
curl -i -s "http://127.0.0.1:8000/features?format=csv"
'''

from fastapi import FastAPI, HTTPException, Header, Query
from pydantic import BaseModel, Field, constr, conint, validator
from typing import List, Optional, Dict, Literal
from pathlib import Path
import os, io, json, time, re, logging, hashlib

import boto3
from botocore.exceptions import BotoCoreError, ClientError

import numpy as np
import pandas as pd
from joblib import load as joblib_load

from fastapi import Response
from typing import Literal

app = FastAPI(title="COVID-Lasso Inference API", version="1.7")

# =========================
# Config
# =========================
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

S3_BUCKET = os.getenv("S3_BUCKET", "ach-covid-lasso-us-east-2")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")

ALLOWED_PRESIGN_METHODS = ("put_object", "get_object")
ALLOWED_S3_PREFIXES = ("uploads/", "results/", "tmp/")
MAX_PRESIGN_SECS = 60 * 60 * 24  # 24h

REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
API_KEY = os.getenv("API_KEY", "")

# Model artifacts
MODEL_VERSION = os.getenv("MODEL_VERSION", "cfr-lasso-v1")
FEATURES_URI = os.getenv("FEATURES_URI", str(ROOT_DIR / "lasso_training_data" / "selected_features.csv"))
SCALER_URI   = os.getenv("SCALER_URI",   str(ROOT_DIR / "model_artifacts" / "scaler.joblib"))
MODEL_URI    = os.getenv("MODEL_URI",    str(ROOT_DIR / "model_artifacts" / "lasso_model.joblib"))
CALIB_URI    = os.getenv("CALIB_URI",    "")  # optional

# =========================
# AWS clients
# =========================
s3 = boto3.client("s3", region_name=AWS_REGION)
sqs = boto3.client("sqs", region_name=AWS_REGION)

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

# =========================
# Schemas
# =========================
class FeaturesPayload(BaseModel):
    sample_id: constr(strip_whitespace=True, min_length=1)
    # Contract: "<REGION>_<POSITION>" (digits only)
    features: Dict[str, conint(ge=0, le=1)] = Field(..., description="Binary features like ORF1ab_10, S_957, N_203")

class TopFeature(BaseModel):
    feature: str
    coef: float                # model coefficient
    contribution: float        # |coef * x_i| for THIS request (in model/standardized space)
    coef_str: str              # scientific-notation string for display
    contribution_str: str      # scientific-notation string for display

class PredictResponse(BaseModel):
    sample_id: str
    cfr_pred: float                   # single source of truth (final value)
    cfr_pred_str: Optional[str] = None  # high-precision display helper
    cfr_pred_pct: Optional[str] = None  # UI convenience (percentage string)
    model: str
    version: str
    top_features: Optional[List[TopFeature]] = None
    feature_file_sha: Optional[str] = None  # provenance for header file

class PresignRequest(BaseModel):
    key: constr(strip_whitespace=True, min_length=3)
    method: Literal["put_object", "get_object"] = "put_object"
    content_type: Optional[str] = "application/octet-stream"
    expires_in: int = 3600

    @validator("expires_in")
    def _cap_exp(cls, v):
        return min(max(1, v), MAX_PRESIGN_SECS)

    @validator("key")
    def _key_prefix(cls, v):
        if ".." in v or v.startswith("/") or "//" in v:
            raise ValueError("invalid key")
        if not v.startswith(ALLOWED_S3_PREFIXES):
            raise ValueError(f"key must start with one of: {ALLOWED_S3_PREFIXES}")
        return v

class PresignResponse(BaseModel):
    bucket: str
    key: str
    url: str
    expires_in: int
    method: Literal["put_object", "get_object"]

class SubmitJob(BaseModel):
    input_s3: constr(strip_whitespace=True)
    output_s3: constr(strip_whitespace=True)
    params: Optional[Dict[str, str]] = None
    idempotency_key: Optional[constr(strip_whitespace=True, min_length=6)] = None

class SubmitResponse(BaseModel):
    job_id: str
    status: Literal["queued"]

# =========================
# Auth helper
# =========================
def require_api_key(x_api_key: Optional[str]):
    if REQUIRE_API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# =========================
# Feature-name contract (strict: <REGION>_<POSITION>, digits only)
# =========================
GENES = ("S","N","M","E","ORF1ab","ORF3a","ORF6","ORF7a","ORF7b","ORF8","ORF10")
_FEATURE_PATTERN = re.compile(rf"^(?:{'|'.join(GENES)})_[0-9]{{1,7}}$")
MAX_INVALID_TO_SHOW = 25

def _validate_feature_structure(payload_keys: Dict[str, int]) -> None:
    invalid = []
    for k in payload_keys.keys():
        if not _FEATURE_PATTERN.match(k):
            invalid.append(k)
            if len(invalid) >= MAX_INVALID_TO_SHOW:
                break
    if invalid:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid feature name format",
                "invalid": invalid,
                "invalid_count": len(invalid),
                "hint": "Use '<REGION>_<POSITION>' e.g., ORF1ab_10, S_957, N_203",
            },
        )

def _warn_unknown_valid_keys(payload_keys: Dict[str,int], feature_order: List[str]) -> None:
    valid_set = set(feature_order)
    unknown_valid = [k for k in payload_keys if k not in valid_set and _FEATURE_PATTERN.match(k)]
    if unknown_valid:
        log.warning({"msg": "unknown_valid_features_ignored", "count": len(unknown_valid), "sample": unknown_valid[:10]})

# =========================
# Artifact loading (local paths for now)
# =========================
def _read_bytes_local(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _load_feature_order(uri: str) -> List[str]:
    raw = _read_bytes_local(uri)
    if uri.endswith(".json"):
        order = json.loads(raw.decode("utf-8"))
        if not isinstance(order, list):
            raise RuntimeError("feature order JSON must be a list")
    elif uri.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw), nrows=0)  # header-only
        order = df.columns.tolist()
    else:
        raise RuntimeError("FEATURES_URI must end with .csv or .json")
    if not order:
        raise RuntimeError("feature order file is empty")
    if len(order) != len(set(order)):
        raise RuntimeError("duplicate feature names in feature order file")
    return order

# Globals
MODEL = None
SCALER = None
CALIB = None
FEATURE_ORDER: List[str] = []
FEATURE_FILE_SHA = ""

@app.on_event("startup")
def _load_artifacts():
    global MODEL, SCALER, CALIB, FEATURE_ORDER, FEATURE_FILE_SHA
    try:
        feat_bytes = _read_bytes_local(FEATURES_URI)
        FEATURE_FILE_SHA = _sha256_hex(feat_bytes)[:12]

        # Enforce expected feature header fingerprint (provenance guard)
        expected_sha = os.getenv("EXPECTED_FEATURE_SHA", "").strip()
        enforce = os.getenv("FEATURE_SHA_ENFORCE", "strict").lower()
        if expected_sha:
            if FEATURE_FILE_SHA != expected_sha:
                msg = (
                    f"Feature SHA mismatch: have={FEATURE_FILE_SHA} expected={expected_sha}. "
                    f"Set EXPECTED_FEATURE_SHA to the correct value or update your feature header."
                )
                if enforce == "strict":
                    log.error(msg)
                    raise RuntimeError(msg)
                else:
                    log.warning("WARN mode: " + msg)
            else:
                log.info(f"Feature SHA matches expected: {expected_sha}")
        else:
            log.info("No EXPECTED_FEATURE_SHA set; skipping feature SHA enforcement.")

        FEATURE_ORDER = _load_feature_order(FEATURES_URI)
        D = len(FEATURE_ORDER)

        SCALER = joblib_load(io.BytesIO(_read_bytes_local(SCALER_URI)))
        MODEL  = joblib_load(io.BytesIO(_read_bytes_local(MODEL_URI)))

        if CALIB_URI:
            try:
                CALIB = joblib_load(io.BytesIO(_read_bytes_local(CALIB_URI)))
                log.info("Calibrator loaded")
            except Exception as e:
                CALIB = None
                log.warning(f"Calibrator not loaded: {e}")

        if hasattr(SCALER, "n_features_in_") and SCALER.n_features_in_ != D:
            raise RuntimeError(f"Scaler expects {SCALER.n_features_in_} features, but header has {D}")
        if hasattr(MODEL, "coef_"):
            model_D = int(np.asarray(MODEL.coef_).ravel().shape[0])
            if model_D != D:
                raise RuntimeError(f"Model expects {model_D} features, but header has {D}")

        log.info(f"Artifacts loaded OK: D={D}, model={type(MODEL).__name__}, version={MODEL_VERSION}, feat_sha={FEATURE_FILE_SHA}")
    except Exception:
        log.exception("Failed to load model/scaler/feature order")
        raise

# =========================
# Inference helpers
# =========================
def _vectorize(features: Dict[str, int]) -> np.ndarray:
    if not FEATURE_ORDER:
        raise RuntimeError("Feature order not loaded")
    x = np.zeros((1, len(FEATURE_ORDER)), dtype=float)
    idx = {f: i for i, f in enumerate(FEATURE_ORDER)}
    for k, v in features.items():
        j = idx.get(k)
        if j is not None:
            x[0, j] = 1.0 if v else 0.0
    return x

def _top_features_linear(model, Xs: np.ndarray, k: int = 10) -> List[TopFeature]:
    """
    Return top-k features that actually contributed for THIS input.
    Sort by |coef * x| in model space. If all contributions are zero,
    fall back to largest |coef| (non-zero).
    """
    if not hasattr(model, "coef_"):
        return []
    coef = np.asarray(model.coef_, dtype=float).ravel()
    xval = Xs.ravel()
    contrib = np.abs(coef * xval)

    nz = np.where(contrib > 0)[0]
    if nz.size:
        idx = nz[np.argsort(contrib[nz])[::-1]][:k]
    else:
        nzcoef = np.where(coef != 0)[0]
        if nzcoef.size == 0:
            return []
        idx = nzcoef[np.argsort(np.abs(coef[nzcoef]))[::-1]][:k]

    out: List[TopFeature] = []
    for i in idx:
        if i >= len(FEATURE_ORDER):
            continue
        c = float(coef[i])
        g = float(contrib[i])
        out.append(
            TopFeature(
                feature=FEATURE_ORDER[i],
                coef=c,
                contribution=g,
                coef_str=f"{c:.12g}",
                contribution_str=f"{g:.12g}",
            )
        )
    return out

# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "time": int(time.time())}

@app.get("/version")
def version():
    return {"model_version": MODEL_VERSION, "feature_file_sha": FEATURE_FILE_SHA}

@app.post("/predict", response_model=PredictResponse)
def predict(
    body: FeaturesPayload,
    x_api_key: Optional[str] = Header(None),
    debug: bool = Query(False),  # keep for future: include extra fields only when needed
):
    require_api_key(x_api_key)
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    _validate_feature_structure(body.features)
    _warn_unknown_valid_keys(body.features, FEATURE_ORDER)

    X = _vectorize(body.features)
    try:
        Xs = SCALER.transform(X)
    except Exception:
        log.exception("scaler.transform failed")
        raise HTTPException(status_code=500, detail="scaler error")

    try:
        y = float(MODEL.predict(Xs)[0])
    except Exception:
        log.exception("model.predict failed")
        raise HTTPException(status_code=500, detail="model inference error")

    if CALIB is not None:
        try:
            y = float(CALIB.predict(np.array([[y]]))[0])  # many calibrators expect (N,1)
        except Exception:
            log.warning("calibration failed; returning uncalibrated value")

    tops = _top_features_linear(MODEL, Xs, k=10)

    resp = PredictResponse(
        sample_id=body.sample_id,
        cfr_pred=y,
        cfr_pred_str=f"{y:.12g}",
        cfr_pred_pct=f"{y*100:.2f}%",
        model=type(MODEL).__name__,
        version=MODEL_VERSION,
        top_features=tops or None,
        feature_file_sha=FEATURE_FILE_SHA,
    )

    # If in future you want to expose additional internals (e.g., raw pre-calibration value),
    # add them conditionally when `debug=True`.

    return resp

@app.get("/features", response_model=None)
def get_features(format: Literal["json", "csv"] = "json"):
    if not FEATURE_ORDER:
        raise HTTPException(status_code=503, detail="feature order not loaded")

    etag = FEATURE_FILE_SHA or hashlib.sha256(",".join(FEATURE_ORDER).encode()).hexdigest()[:12]

    if format == "csv":
        csv_line = ",".join(FEATURE_ORDER) + "\n"
        headers = {
            "ETag": etag,
            "Cache-Control": "public, max-age=300",
            "Content-Disposition": 'attachment; filename="selected_features.header.csv"',
        }
        return Response(content=csv_line, media_type="text/csv", headers=headers)

    # JSON (default)
    return {
        "sha": etag,
        "count": len(FEATURE_ORDER),
        "features": FEATURE_ORDER,
    }

@app.post("/presign", response_model=PresignResponse)
def presign(req: PresignRequest, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    try:
        if req.method == "put_object":
            url = s3.generate_presigned_url(
                ClientMethod="put_object",
                Params={"Bucket": S3_BUCKET, "Key": req.key, "ContentType": req.content_type},
                ExpiresIn=req.expires_in
            )
        else:
            url = s3.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": S3_BUCKET, "Key": req.key},
                ExpiresIn=req.expires_in
            )
        return PresignResponse(bucket=S3_BUCKET, key=req.key, url=url, expires_in=req.expires_in, method=req.method)
    except (BotoCoreError, ClientError):
        log.exception("presign failed")
        raise HTTPException(status_code=502, detail="presign error")

@app.post("/submit", response_model=SubmitResponse)
def submit(job: SubmitJob, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    if not SQS_QUEUE_URL:
        raise HTTPException(status_code=500, detail="SQS not configured")

    msg = {
        "input_s3": job.input_s3,
        "output_s3": job.output_s3,
        "params": job.params or {},
        "requested_at": int(time.time()),
    }
    try:
        kwargs = dict(QueueUrl=SQS_QUEUE_URL, MessageBody=json.dumps(msg))
        if job.idempotency_key:
            kwargs["MessageAttributes"] = {
                "IdempotencyKey": {"DataType": "String", "StringValue": job.idempotency_key}
            }
        resp = sqs.send_message(**kwargs)
        return SubmitResponse(job_id=resp.get("MessageId"), status="queued")
    except (BotoCoreError, ClientError):
        log.exception("submit failed")
        raise HTTPException(status_code=502, detail="queue error")
