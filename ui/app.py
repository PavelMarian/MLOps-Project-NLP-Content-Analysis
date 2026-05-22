"""
MLOps Content Moderation - Web UI
FastAPI приложение для мониторинга, инференса и управления моделью
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import datetime
import random
import time
import uuid

app = FastAPI(title="MLOps Content Moderation UI", version="1.0.0")
app.mount("/static", StaticFiles(directory="ui/static"), name="static")
templates = Jinja2Templates(directory="ui/templates")

class PredictRequest(BaseModel):
    text: str
    model_version: str = "latest"

# Заглушечные данные
predictions_store = []
system_metrics = {"total_processed": 15234, "avg_latency_ms": 187, "uptime_hours": 168, "rps": 3.2}
quality_metrics = {"accuracy": 0.892, "precision": 0.876, "recall": 0.901, "f1": 0.888}
drift_metrics = {"data_drift": 0.05, "concept_drift": 0.03, "target_drift": 0.02}
experiments_store = [
    {"version": "v1.0", "timestamp": "2025-04-01T10:00:00", "metrics": {"accuracy": 0.85, "f1": 0.84}, "status": "completed"},
    {"version": "v1.1", "timestamp": "2025-04-10T14:30:00", "metrics": {"accuracy": 0.87, "f1": 0.86}, "status": "completed"},
    {"version": "v2.0", "timestamp": "2025-04-20T09:15:00", "metrics": {"accuracy": 0.89, "f1": 0.89}, "status": "completed"},
    {"version": "v2.1", "timestamp": datetime.datetime.now().isoformat(), "metrics": {"accuracy": 0.892, "f1": 0.888}, "status": "production"}
]
training_data = {"last_training": "2025-04-20T09:15:00", "training_samples": 50000, "is_training": False}

def generate_mock_prediction(text: str, model_version: str):
    start = time.time()
    toxic_keywords = ["дурак", "идиот", "тупой", "убить", "ненавижу", "сволочь", "дебил", "скотина", "тварь"]
    text_lower = text.lower()
    toxic_score = sum(1 for kw in toxic_keywords if kw in text_lower) / len(toxic_keywords)
    is_toxic = toxic_score > 0.1 or "токсично" in text_lower
    features = {"toxicity_score": min(toxic_score + random.uniform(-0.1,0.1), 1.0), "hate_score": random.uniform(0,0.3), "threat_score": random.uniform(0,0.2), "profanity_score": toxic_score}
    return {"is_toxic": is_toxic, "confidence": toxic_score if is_toxic else 1-toxic_score, "model_version": model_version, "processing_time_ms": round((time.time()-start)*1000, 2), "features": features}

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "active_page": "dashboard"})

@app.get("/inference", response_class=HTMLResponse)
async def inference_page(request: Request):
    return templates.TemplateResponse("inference.html", {"request": request, "active_page": "inference"})

@app.get("/experiments", response_class=HTMLResponse)
async def experiments_page(request: Request):
    return templates.TemplateResponse("experiments.html", {"request": request, "active_page": "experiments"})

@app.post("/api/predict")
async def predict(req: PredictRequest):
    pred = generate_mock_prediction(req.text, req.model_version)
    predictions_store.append({"id": str(uuid.uuid4()), "timestamp": datetime.datetime.now().isoformat(), "text": req.text, "is_toxic": pred["is_toxic"], "confidence": pred["confidence"], "model_version": pred["model_version"]})
    if len(predictions_store) > 100: predictions_store.pop(0)
    system_metrics["total_processed"] += 1
    system_metrics["avg_latency_ms"] = system_metrics["avg_latency_ms"] * 0.99 + pred["processing_time_ms"] * 0.01
    return pred

@app.get("/api/metrics")
async def get_metrics():
    return {"system": system_metrics, "quality": quality_metrics, "drift": drift_metrics}

@app.get("/api/predictions/latest")
async def get_latest_predictions():
    return list(reversed(predictions_store[-20:]))

@app.post("/api/retrain")
async def retrain():
    training_data["is_training"] = True
    # симуляция переобучения
    quality_metrics["accuracy"] = min(0.95, quality_metrics["accuracy"] + 0.005)
    drift_metrics["data_drift"] = max(0, drift_metrics["data_drift"] - 0.01)
    training_data["is_training"] = False
    training_data["last_training"] = datetime.datetime.now().isoformat()
    return {"message": "Переобучение запущено успешно", "status": "started"}

@app.get("/api/experiments")
async def get_experiments():
    return {"experiments": experiments_store, "latest_version": experiments_store[-1]["version"]}

@app.get("/api/history")
async def get_history():
    """История метрик для графиков (заглушка)"""
    import random
    days = list(range(7, 0, -1))
    quality_history = []
    drift_history = []
    base_acc = 0.88
    for i, day in enumerate(days):
        quality_history.append({
            "date": f"День {day}",
            "accuracy": base_acc + random.uniform(-0.02, 0.02),
            "precision": 0.87 + random.uniform(-0.02, 0.02),
            "recall": 0.89 + random.uniform(-0.02, 0.02),
            "f1": 0.88 + random.uniform(-0.02, 0.02)
        })
        drift_history.append({
            "date": f"День {day}",
            "data_drift": max(0, 0.03 + random.uniform(-0.02, 0.04)),
            "concept_drift": max(0, 0.02 + random.uniform(-0.01, 0.03)),
            "target_drift": max(0, 0.01 + random.uniform(-0.01, 0.02))
        })
    return {"quality": quality_history, "drift": drift_history}
