import os
import secrets
from contextlib import asynccontextmanager

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    status,
)
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
    return (
        app.state.predictor
        .predict(request.text)
    )
