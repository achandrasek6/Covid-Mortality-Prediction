from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Request
from pydantic import BaseModel, Field, constr, conint, validator
from typing import List, Optional, Dict, Literal
import os, json, boto3, time, re, logging
from botocore.exceptions import BotoCoreError, ClientError

app = FastAPI(title="COVID-Lasso Inference API", version="1.1")

# --- Config ---
S3_BUCKET = os.getenv("S3_BUCKET", "ach-covid-lasso-us-east-2")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
ALLOWED_PRESIGN_METHODS = ("put_object", "get_object")
ALLOWED_S3_PREFIXES = ("uploads/", "results/", "tmp/")  # adjust for your usage
MAX_PRESIGN_SECS = 60 * 60 * 24  # 24h cap
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
API_KEY = os.getenv("API_KEY", "")  # set in ECS task if using

# --- AWS clients ---
s3 = boto3.client("s3", region_name=AWS_REGION)
sqs = boto3.client("sqs", region_name=AWS_REGION)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

# --- Models ---
class FeaturesPayload(BaseModel):
    sample_id: constr(strip_whitespace=True, min_length=1)
    # binary features only (0/1)
    features: Dict[str, conint(ge=0, le=1)] = Field(..., description="Binary mutation features (e.g., S_N501Y:1)")

class PredictResponse(BaseModel):
    sample_id: str
    cfr_pred: float
    model: str
    version: str

class PresignRequest(BaseModel):
    key: constr(strip_whitespace=True, min_length=3)
    method: Literal["put_object", "get_object"] = "put_object"
    content_type: Optional[str] = "application/octet-stream"
    expires_in: int = 3600

    @validator("expires_in")
    def _cap_exp(cls, v):
        return min(v, MAX_PRESIGN_SECS)

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

# --- Simple API-key guard (optional) ---
def require_api_key(x_api_key: Optional[str]):
    if REQUIRE_API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# --- Startup: load model/scaler once ---
MODEL = None
SCALER = None
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

@app.on_event("startup")
def _load_model():
    global MODEL, SCALER
    try:
        # TODO: replace with actual joblib/pickle loads
        # from joblib import load
        # SCALER = load("/models/scaler.joblib")
        # MODEL = load("/models/lasso_model.joblib")
        MODEL = "lasso_stub"
        SCALER = "scaler_stub"
        log.info("Model/scaler loaded OK")
    except Exception as e:
        log.exception("Failed to load model artifacts")
        raise

# --- Helpers ---
def _vectorize(features: Dict[str, int]) -> list:
    # TODO: vectorization respecting training order; placeholder keeps API shape
    return [features.get(k, 0) for k in sorted(features.keys())]

# --- Routes ---
@app.get("/health")
def health():
    return {"status": "ok", "time": int(time.time())}

@app.post("/predict", response_model=PredictResponse)
def predict(body: FeaturesPayload, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    # TODO: use real scaler.transform & model.predict
    # vec = SCALER.transform([_vectorize(body.features)])
    # score = float(MODEL.predict(vec)[0])
    score = 0.01234  # placeholder
    return PredictResponse(
        sample_id=body.sample_id,
        cfr_pred=score,
        model="lasso_baseline",
        version=MODEL_VERSION,
    )

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
        else:  # get_object
            url = s3.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": S3_BUCKET, "Key": req.key},
                ExpiresIn=req.expires_in
            )
        return PresignResponse(bucket=S3_BUCKET, key=req.key, url=url, expires_in=req.expires_in, method=req.method)
    except (BotoCoreError, ClientError) as e:
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
        "requested_at": int(time.time())
    }
    try:
        kwargs = dict(QueueUrl=SQS_QUEUE_URL, MessageBody=json.dumps(msg))
        if job.idempotency_key:
            # for FIFO queues, also set MessageGroupId/MessageDeduplicationId
            kwargs["MessageAttributes"] = {
                "IdempotencyKey": {"DataType": "String", "StringValue": job.idempotency_key}
            }
        resp = sqs.send_message(**kwargs)
        return SubmitResponse(job_id=resp.get("MessageId"), status="queued")
    except (BotoCoreError, ClientError):
        log.exception("submit failed")
        raise HTTPException(status_code=502, detail="queue error")
