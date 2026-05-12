import os
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.bert_classifier import ToxicityClassifier

MODEL_PATH = os.getenv("MODEL_PATH", "models/finetuned")  # путь к папке с моделью
DEVICE = os.getenv("DEVICE", None)  # "cuda" или None (auto)
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

classifier = ToxicityClassifier(model_path=MODEL_PATH, device=DEVICE, threshold=THRESHOLD)

class PredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="Список текстов для анализа")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Порог токсичности")

class PredictionResult(BaseModel):
    text: str
    cleaned_text: str
    toxic_prob: float
    label: str
    confidence: float

class PredictResponse(BaseModel):
    results: List[PredictionResult]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

app = FastAPI(
    title="Toxicity Detection API",
    description="Детектор токсичности на основе BERT",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc",    # ReDoc
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=classifier is not None,
        device=classifier.device
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        threshold = request.threshold
        results = classifier.predict(request.texts, threshold=threshold)
        return PredictResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)