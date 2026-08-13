import os

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.inference import get_predictor


# ============================================================
# Configuration
# ============================================================

MODEL_BACKEND = os.getenv(
    "MODEL_BACKEND",
    "baseline",
)


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print(
        f"Loading model: {MODEL_BACKEND}"
    )

    app.state.predictor = (
        get_predictor(
            MODEL_BACKEND
        )
    )

    yield

    app.state.predictor = None


# ============================================================
# Application
# ============================================================

app = FastAPI(
    title=(
        "IT Service Ticket Classifier"
    ),

    description=(
        "Automatically classifies "
        "IT service tickets into "
        "support categories."
    ),

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
        examples=[
            (
                "I cannot access "
                "the shared network drive"
            )
        ],
    )


class TopPrediction(BaseModel):

    category: str

    probability: float


class PredictionResponse(BaseModel):

    category: str

    confidence: float

    threshold: float

    needs_manual_review: bool

    top_3: list[
        TopPrediction
    ]


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():

    return {
        "message":
            "IT Service Ticket Classifier"
    }


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",

        "model_backend":
            MODEL_BACKEND,
    }


# ============================================================
# Prediction
# ============================================================

@app.post(
    "/predict",
    response_model=
        PredictionResponse,
)
def predict_ticket(
    request: TicketRequest,
):

    result = (
        app.state
        .predictor
        .predict(
            request.text
        )
    )

    return result