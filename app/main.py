import os
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")

MODEL_NAME: str = os.environ.get("MODEL_NAME", "")
MODEL_PATH: str = os.environ.get("MODEL_PATH", "")

_model: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _model
    if not MODEL_PATH:
        raise RuntimeError("MODEL_PATH environment variable is not set")
    logger.info("Loading model from %s", MODEL_PATH)
    _model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
    yield
    _model = None


app = FastAPI(title="Loan Status Prediction API", version="1.0.0", lifespan=lifespan)


class PredictionRequest(BaseModel):
    features: dict[str, float]


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        df = pd.DataFrame([request.features])
        prediction = int(_model.predict(df)[0])
        probability = float(_model.predict_proba(df)[0][prediction])
        label = "Fully Paid" if prediction == 1 else "Charge Off"
        return PredictionResponse(
            prediction=prediction,
            prediction_label=label,
            probability=probability,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
