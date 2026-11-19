import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.predictor import Predictor

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = BASE_DIR / "models" / "ddos_model.joblib"
INDEX_HTML = Path(__file__).resolve().parent / "static" / "index.html"

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = Predictor(os.getenv("MODEL_PATH", str(DEFAULT_MODEL)))
    yield


app = FastAPI(title="DDoS Detector", lifespan=lifespan)


class FlowInput(BaseModel):
    features: dict[str, float]


def get_predictor() -> Predictor:
    predictor = state.get("predictor")
    if predictor is None or not predictor.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")
    return predictor


@app.get("/")
def home():
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health():
    predictor = state.get("predictor")
    return {"status": "ok", "model_loaded": bool(predictor and predictor.loaded)}


@app.get("/features")
def features():
    p = get_predictor()
    return {"features": p.features, "examples": p.examples, "metrics": p.metrics}


@app.post("/predict")
def predict(body: FlowInput):
    p = get_predictor()
    try:
        return p.predict(body.features)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))
