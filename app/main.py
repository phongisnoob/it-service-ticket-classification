import os
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field, field_validator
from starlette.routing import Match

from src.inference import PredictionResult, get_predictor

APP_ENV = os.getenv("APP_ENV", "development").lower()
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "auto")

# Fixed label for unmatched routes so arbitrary URLs cannot create unbounded cardinality.
_UNMATCHED_ROUTE = "UNMATCHED"

HTTP_REQUESTS = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"]
)
PREDICTION_REQUESTS = Counter("prediction_requests_total", "Total prediction requests")
AUTO_ROUTE_COUNT = Counter("auto_route_total", "Total auto-routed predictions")
MANUAL_REVIEW_COUNT = Counter("manual_review_total", "Total predictions needing manual review")
PREDICTION_CONFIDENCE = Histogram("prediction_confidence", "Prediction confidence score")


def _resolve_route_template(request: Request) -> str:
    """Return the matched route template or a fixed sentinel for unmatched routes."""
    for route in request.app.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            path: str = getattr(route, "path", _UNMATCHED_ROUTE)
            return path
    return _UNMATCHED_ROUTE


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    api_key = os.getenv("API_KEY")
    if APP_ENV == "production" and not api_key:
        raise RuntimeError(
            "API_KEY must be set when APP_ENV=production. "
            "Set the variable or switch to APP_ENV=development for local/test use."
        )
    print(f"Loading model: {MODEL_BACKEND} (APP_ENV={APP_ENV})")
    app.state.predictor = get_predictor(MODEL_BACKEND)
    yield
    app.state.predictor = None


app = FastAPI(
    title="IT Service Ticket Classifier",
    description="Automatically classifies IT service tickets into support categories.",
    version="1.0.0",
    lifespan=lifespan,
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.middleware("http")
async def monitor_requests(request: Request, call_next: Any) -> Any:
    start_time = time.time()
    endpoint = _resolve_route_template(request)
    response: Response | None = None
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        raise
    finally:
        duration = time.time() - start_time
        if endpoint != "/metrics":
            HTTP_REQUESTS.labels(
                method=request.method, endpoint=endpoint, status=status_code
            ).inc()
            HTTP_LATENCY.labels(method=request.method, endpoint=endpoint).observe(duration)


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
            raise ValueError("Ticket text must contain at least 3 non-whitespace characters.")
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


API_KEY = os.getenv("API_KEY")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    # When API_KEY is unset and not in production, the service runs in open/demo mode.
    if API_KEY is None:
        return
    if x_api_key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "IT Service Ticket Classifier"}


@app.get("/health")
def health() -> dict[str, object]:
    predictor = app.state.predictor
    return {
        "status": "ok",
        "model_backend": getattr(predictor, "backend", MODEL_BACKEND),
        "model_sha256": getattr(predictor, "model_sha256", None),
        "threshold": getattr(predictor, "threshold", None),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_ticket(
    request: TicketRequest,
    _: None = Depends(require_api_key),
) -> PredictionResult:
    PREDICTION_REQUESTS.inc()
    result: PredictionResult = app.state.predictor.predict(request.text)

    PREDICTION_CONFIDENCE.observe(result["confidence"])

    if result["needs_manual_review"]:
        MANUAL_REVIEW_COUNT.inc()
    else:
        AUTO_ROUTE_COUNT.inc()

    return result
