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
curl -sS -X POST $BASE/predict   -H 'Content-Type: application/json'   --data-binary @/tmp/push_up.json | jq .
curl -sS -X POST $BASE/predict   -H 'Content-Type: application/json'   --data-binary @/tmp/push_down.json | jq .

Feature-space pinning demo
--------------------------
# 1) Find the observed SHA:
curl -s $BASE/version
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
curl -s $BASE/features | jq

# CSV (header-only, with download filename)
curl -i -s "$BASE/features?format=csv"

# For local use, replace $BASE with "http://localhost:8000"
'''

from fastapi import FastAPI, HTTPException, Header, Query, Response
from pydantic import BaseModel, Field, constr, conint
try:
    # pydantic v2
    from pydantic import field_validator as validator
except Exception:
    # pydantic v1 fallback
    from pydantic import validator  # type: ignore

from typing import List, Optional, Dict, Literal, Any
from pathlib import Path
from datetime import datetime, timezone
import os, io, json, time, re, logging, hashlib, uuid

import boto3
from botocore.exceptions import BotoCoreError, ClientError

import numpy as np
import pandas as pd
from joblib import load as joblib_load

app = FastAPI(title="COVID-Lasso Inference API", version="1.8")

# =========================
# Config / Env
# =========================
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

AWS_REGION     = os.getenv("AWS_REGION", "us-east-2")
UPLOADS_BUCKET = os.getenv("UPLOADS_BUCKET", os.getenv("S3_BUCKET", ""))   # legacy fallback to S3_BUCKET
RESULTS_BUCKET = os.getenv("RESULTS_BUCKET", os.getenv("S3_BUCKET", ""))   # legacy fallback
SQS_QUEUE_URL  = os.getenv("JOBS_QUEUE_URL") or os.getenv("SQS_QUEUE_URL", "")
JOBS_TABLE     = os.getenv("JOBS_TABLE", "covid_cfr_jobs")

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
s3   = boto3.client("s3", region_name=AWS_REGION)
sqs  = boto3.client("sqs", region_name=AWS_REGION)
dyna = boto3.resource("dynamodb", region_name=AWS_REGION).Table(JOBS_TABLE)

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

# =========================
# Schemas (inference)
# =========================
class FeaturesPayload(BaseModel):
    sample_id: constr(strip_whitespace=True, min_length=1)
    features: Dict[str, conint(ge=0, le=1)] = Field(..., description="Binary features like ORF1ab_10, S_957, N_203")

class TopFeature(BaseModel):
    feature: str
    coef: float
    contribution: float
    coef_str: str
    contribution_str: str

class PredictResponse(BaseModel):
    sample_id: str
    cfr_pred: float
    cfr_pred_str: Optional[str] = None
    cfr_pred_pct: Optional[str] = None
    model: str
    version: str
    top_features: Optional[List[TopFeature]] = None
    feature_file_sha: Optional[str] = None

# =========================
# Schemas (jobs & presign)
# =========================
class PresignRequest(BaseModel):
    key: constr(strip_whitespace=True, min_length=3)
    method: Literal["put_object", "get_object"] = "put_object"
    content_type: Optional[str] = "application/octet-stream"
    expires_in: int = 900

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
    input_s3: constr(strip_whitespace=True)    # e.g. f"s3://{UPLOADS_BUCKET}/uploads/<RUN_ID>/"
    output_s3: constr(strip_whitespace=True)   # e.g. f"s3://{RESULTS_BUCKET}/results/<RUN_ID>/"
    params: Dict[str, Any] = {}
    idempotency_key: Optional[constr(strip_whitespace=True, min_length=6)] = None

class SubmitResponse(BaseModel):
    job_id: str
    status: Literal["QUEUED"]

class StatusResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None
    artifacts: Optional[List[Dict[str, str]]] = None  # [{name,url}]

# =========================
# Auth helper
# =========================
def require_api_key(x_api_key: Optional[str]):
    if REQUIRE_API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# =========================
# Feature-name contract
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
# Artifact loading (local paths)
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
# Small utils for jobs
# =========================
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _new_job_id() -> str:
    return str(uuid.uuid4())

def _bucket_for_key(key: str) -> str:
    if key.startswith("uploads/"):
        if not UPLOADS_BUCKET:
            raise HTTPException(500, "UPLOADS_BUCKET not configured")
        return UPLOADS_BUCKET
    if key.startswith("results/") or key.startswith("tmp/"):
        if not RESULTS_BUCKET:
            raise HTTPException(500, "RESULTS_BUCKET not configured")
        return RESULTS_BUCKET
    raise HTTPException(400, f"key must start with one of: {ALLOWED_S3_PREFIXES}")

def _list_results_and_presign(output_s3: str, expires=900) -> List[Dict[str, str]]:
    # output_s3 like s3://bucket/results/<RUN_ID>/
    if not output_s3.startswith("s3://"):
        return []
    _, rest = output_s3.split("s3://", 1)
    bucket, prefix = rest.split("/", 1)
    if not prefix.endswith("/"):
        prefix += "/"

    artifacts: List[Dict[str, str]] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            url = s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires)
            artifacts.append({"name": key.rsplit("/", 1)[-1], "url": url})
    return artifacts

# =========================
# Routes — health/version
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "time": int(time.time())}

@app.get("/version")
def version():
    return {"model_version": MODEL_VERSION, "feature_file_sha": FEATURE_FILE_SHA}

# =========================
# Routes — inference
# =========================
@app.post("/predict", response_model=PredictResponse)
def predict(
    body: FeaturesPayload,
    x_api_key: Optional[str] = Header(None),
    debug: bool = Query(False),
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
            y = float(CALIB.predict(np.array([[y]]))[0])
        except Exception:
            log.warning("calibration failed; returning uncalibrated value")

    tops = _top_features_linear(MODEL, Xs, k=10)

    return PredictResponse(
        sample_id=body.sample_id,
        cfr_pred=y,
        cfr_pred_str=f"{y:.12g}",
        cfr_pred_pct=f"{y*100:.2f}%",
        model=type(MODEL).__name__,
        version=MODEL_VERSION,
        top_features=tops or None,
        feature_file_sha=FEATURE_FILE_SHA,
    )

@app.get("/features")
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

    return {"sha": etag, "count": len(FEATURE_ORDER), "features": FEATURE_ORDER}

# =========================
# Routes — presign / submit / status
# =========================
@app.post("/presign", response_model=PresignResponse)
def presign(req: PresignRequest, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)

    # safety: only allow PUTs under uploads/
    if req.method == "put_object" and not req.key.startswith("uploads/"):
        raise HTTPException(status_code=400, detail="PUTs must be under uploads/<RUN_ID>/...")

    bucket = _bucket_for_key(req.key)
    try:
        if req.method == "put_object":
            url = s3.generate_presigned_url(
                ClientMethod="put_object",
                Params={"Bucket": bucket, "Key": req.key, "ContentType": req.content_type},
                ExpiresIn=req.expires_in
            )
        else:
            url = s3.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": bucket, "Key": req.key},
                ExpiresIn=req.expires_in
            )
        return PresignResponse(bucket=bucket, key=req.key, url=url, expires_in=req.expires_in, method=req.method)
    except (BotoCoreError, ClientError):
        log.exception("presign failed")
        raise HTTPException(status_code=502, detail="presign error")

@app.post("/submit", response_model=SubmitResponse, status_code=202)
def submit(job: SubmitJob, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    if not SQS_QUEUE_URL:
        raise HTTPException(status_code=500, detail="SQS not configured")
    if not (job.input_s3.startswith("s3://") and job.output_s3.startswith("s3://")):
        raise HTTPException(400, "input_s3/output_s3 must be s3://...")

    job_id = job.idempotency_key or _new_job_id()
    item = {
        "job_id": job_id,
        "status": "QUEUED",
        "input_s3": job.input_s3,
        "output_s3": job.output_s3,
        "params": job.params or {},
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }

    # Idempotent upsert: let it succeed if item missing OR existing status is one of allowed
    try:
        dyna.put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(job_id) OR #s IN (:q,:r,:f)",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":q": "QUEUED", ":r": "RUNNING", ":f": "FAILED"},
        )
    except ClientError as e:
        # If ConditionalCheckFailed, we’ll still enqueue using same job_id to be safe
        if e.response.get("Error", {}).get("Code") != "ConditionalCheckFailedException":
            log.exception("Dynamo put_item failed")
            raise HTTPException(502, "job registry error")

    msg_body = {
        "schema_version": 1,
        "job_id": job_id,
        "input_s3": job.input_s3,
        "output_s3": job.output_s3,
        "params": job.params or {},
        "enqueued_at": _now_iso(),
    }

    try:
        kwargs = {"QueueUrl": SQS_QUEUE_URL, "MessageBody": json.dumps(msg_body)}
        # Only set MessageGroupId for FIFO queues; omit otherwise
        if SQS_QUEUE_URL.endswith(".fifo"):
            kwargs["MessageGroupId"] = str(hash(job_id) % (10**12))
        sqs.send_message(**kwargs)
    except (BotoCoreError, ClientError):
        log.exception("SQS send_message failed")
        raise HTTPException(status_code=502, detail="queue error")

    return SubmitResponse(job_id=job_id, status="QUEUED")

@app.get("/status", response_model=StatusResponse)
def status(job_id: str, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    try:
        r = dyna.get_item(Key={"job_id": job_id})
    except (BotoCoreError, ClientError):
        log.exception("Dynamo get_item failed")
        raise HTTPException(502, "job registry read error")

    if "Item" not in r:
        raise HTTPException(404, f"job_id {job_id} not found")
    item = r["Item"]

    resp = StatusResponse(job_id=job_id, status=item["status"], message=item.get("message"))
    if item["status"] == "SUCCEEDED":
        try:
            resp.artifacts = _list_results_and_presign(item["output_s3"])
        except Exception:
            log.exception("listing artifacts failed")
            # still return status without artifacts
    return resp
