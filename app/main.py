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
_bundle: dict[str, Any] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _model, _bundle
    if not MODEL_PATH:
        raise RuntimeError("MODEL_PATH environment variable is not set")
    logger.info("Loading model from %s", MODEL_PATH)
    loaded = joblib.load(MODEL_PATH)
    if not (isinstance(loaded, dict) and "model" in loaded and "selector" in loaded):
        raise RuntimeError(
            "MODEL_PATH must point to an inference bundle containing model, selector, feature_order, feature_min, and feature_max"
        )
    _bundle = loaded
    _model = loaded["model"]
    logger.info("Loaded inference bundle with selector and normalization metadata")
    logger.info("Model loaded successfully")
    yield
    _model = None
    _bundle = None


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
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Inference bundle not loaded")
    try:
        df = pd.DataFrame([request.features])

        feature_order = _bundle["feature_order"]
        missing_features = [f for f in feature_order if f not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required features: {missing_features}",
            )

        # Always use training feature order to keep shape consistent, ignoring extras.
        x = df.reindex(columns=feature_order).astype(float)
        mins = pd.Series(_bundle["feature_min"], dtype=float).reindex(feature_order)
        maxs = pd.Series(_bundle["feature_max"], dtype=float).reindex(feature_order)
        denom = (maxs - mins).replace(0, 1.0)
        x_norm = ((x - mins) / denom).fillna(0.0)
        x_selected = _bundle["selector"].transform(x_norm)

        prediction = int(_model.predict(x_selected)[0])
        probability = float(_model.predict_proba(x_selected)[0][prediction])

        label = "Fully Paid" if prediction == 1 else "Charge Off"
        return PredictionResponse(
            prediction=prediction,
            prediction_label=label,
            probability=probability,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
