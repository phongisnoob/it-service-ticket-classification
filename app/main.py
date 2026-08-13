import os
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    status,
)
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field, field_validator

from src.inference import get_predictor

# ============================================================
# Configuration
# ============================================================

MODEL_BACKEND = os.getenv(
    "MODEL_BACKEND",
    "auto",
)

# ============================================================
# Metrics
# ============================================================

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"]
)

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

AUTO_ROUTE_COUNT = Counter(
    "auto_route_total",
    "Total auto-routed predictions"
)

MANUAL_REVIEW_COUNT = Counter(
    "manual_review_total",
    "Total predictions needing manual review"
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Prediction confidence score"
)


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading model: {MODEL_BACKEND}")
    app.state.predictor = get_predictor(MODEL_BACKEND)
    yield
    app.state.predictor = None


# ============================================================
# Application
# ============================================================

app = FastAPI(
    title="IT Service Ticket Classifier",
    description="Automatically classifies IT service tickets into support categories.",
    version="1.0.0",
    lifespan=lifespan,
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    if request.url.path != "/metrics":
        HTTP_REQUESTS.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        HTTP_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)

    return response


# ============================================================
# Schemas
# ============================================================

class TicketRequest(BaseModel):
    text: str = Field(
        min_length=3,
        max_length=5000,
        examples=["I cannot access the shared network drive"],
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError(
                "Ticket text must contain at least 3 non-whitespace characters."
            )
        return value


class TopPrediction(BaseModel):
    category: str
    probability: float


class PredictionResponse(BaseModel):
    category: str
    confidence: float
    threshold: float
    needs_manual_review: bool
    top_3: list[TopPrediction]


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():
    return {"message": "IT Service Ticket Classifier"}


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():
    predictor = app.state.predictor
    return {
        "status": "ok",
        "model_backend": getattr(
            predictor,
            "backend",
            MODEL_BACKEND,
        ),
        "model_sha256": getattr(predictor, "model_sha256", None),
        "threshold": getattr(predictor, "threshold", None),
    }


# ============================================================
# Prediction
# ============================================================

API_KEY = os.getenv(
    "API_KEY"
)


def require_api_key(
    x_api_key: str | None = Header(
        default=None
    ),
):
    # Local/demo mode:
    # no API_KEY environment variable means no auth.
    if API_KEY is None:
        return

    if (
        x_api_key is None
        or not secrets.compare_digest(
            x_api_key,
            API_KEY,
        )
    ):
        raise HTTPException(
            status_code=
                status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_ticket(
    request: TicketRequest,
    _=Depends(require_api_key),
):
    PREDICTION_REQUESTS.inc()
    result = app.state.predictor.predict(request.text)

    PREDICTION_CONFIDENCE.observe(result["confidence"])

    if result["needs_manual_review"]:
        MANUAL_REVIEW_COUNT.inc()
    else:
        AUTO_ROUTE_COUNT.inc()

    return result
