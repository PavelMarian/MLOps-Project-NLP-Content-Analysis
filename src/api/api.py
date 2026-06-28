from fastapi import FastAPI, BackgroundTasks, HTTPException, BackgroundTasks, Request
from fastapi.responses import Response, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from collections import deque
import logging
import time
import sys
from pathlib import Path
import json
import httpx

from db import Session, Prediction, ModelVersion, DriftLog, ReferenceData

sys.path.append(str(Path(__file__).parent.parent))

from models.model import create_model_with_mlflow
from drift_router import set_model
# from drift_monitoring.drift_detector import DriftDetector, DriftAnalyzer, metrics

from drift_monitoring.drift_detector import DriftDetector
from drift_monitoring.prometheus_client import get_metrics
from drift_router import router as drift_router, get_drift_status, get_model, set_model
from load_new_data import router as new_data_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Toxicity Detection MLOps API",
    description="API для определения токсичности текстов",
    version="1.0.0"
)

app.include_router(drift_router)
app.include_router(new_data_loader)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

model = None
drift_detector = None
prediction_history = deque(maxlen=1000)

MODEL_PATH = "s-nlp/russian_toxicity_classifier"
DEFAULT_THRESHOLD = 0.5
USE_MLFLOW = True
EXPERIMENT_NAME = "toxicity_model"


class PredictRequest(BaseModel):
    texts: List[str]
    threshold: Optional[float] = None


class PredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    latency_ms: float


class FeedbackRequest(BaseModel):
    prediction_id: int
    is_correct: bool


class FeedbackResponse(BaseModel):
    status: str
    feedback_received: int


class RetrainRequest(BaseModel):
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    output_dir: str = "models/finetuned"
    days_back: int = 30


class RetrainResponse(BaseModel):
    success: bool
    message: str
    f1_score: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None


class DriftResponse(BaseModel):
    drift_detected: bool
    drift_score: float
    last_check: Optional[str] = None


class StatsResponse(BaseModel):
    total_predictions: int
    feedback_received: int
    accuracy_from_feedback: float
    toxic_rate: float
    predictions_24h: int
    feedback_24h: int
    active_model: Optional[Dict[str, Any]] = None
    recent_predictions: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str] = None
    drift_detector: bool
    device: Optional[str] = None
    total_predictions: int


def init_model() -> bool:
    global model

    try:
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = create_model_with_mlflow(
            use_mlflow=USE_MLFLOW,
            experiment_name=EXPERIMENT_NAME
        )
        set_model(model)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def init_drift_detector() -> bool:
    global drift_detector
    session = Session()
    try:
        recent = session.query(Prediction).filter(
            Prediction.feedback.isnot(None)
        ).order_by(
            Prediction.timestamp.desc()
        ).limit(500).all()

        if len(recent) > 100:
            texts = [p.text for p in recent]
            drift_detector = DriftDetector(texts)
            logger.info(f"Drift detector initialized with {len(texts)} samples")
            return True
        logger.info("Not enough data for drift detector")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize drift detector: {e}")
        return False
    finally:
        session.close()


@app.on_event("startup")
async def startup():
    if not init_model():
        logger.error("Failed to initialize model on startup")
    init_drift_detector()


@app.on_event("shutdown")
async def shutdown():
    global model
    if model:
        try:
            model.end_mlflow_run()
        except:
            pass


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        threshold = request.threshold or DEFAULT_THRESHOLD

        results = model.predict(request.texts, threshold=threshold)

        if not isinstance(results, list):
            results = [results]

        session = Session()
        saved_results = []
        try:
            for result in results:
                pred = Prediction(
                    text=result["text"],
                    cleaned_text=result.get("cleaned_text", result["text"]),
                    toxic_prob=result["toxic_prob"],
                    label=result["label"],
                    confidence=result.get("confidence", max(result["toxic_prob"], 1 - result["toxic_prob"])),
                    threshold_used=threshold
                )
                session.add(pred)
                session.flush()
                result["id"] = pred.id
                saved_results.append(result)
                prediction_history.append({"text": result["text"], "id": pred.id})

            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            session.close()

        latency = (time.time() - start_time) * 1000

        background_tasks.add_task(check_drift)

        return PredictResponse(
            predictions=saved_results,
            latency_ms=round(latency, 2)
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback", response_model=FeedbackResponse)
async def add_feedback(feedback: FeedbackRequest, background_tasks: BackgroundTasks):
    session = Session()
    try:
        pred = session.query(Prediction).filter(Prediction.id == feedback.prediction_id).first()
        if not pred:
            raise HTTPException(status_code=404, detail="Prediction not found")

        pred.feedback = feedback.is_correct
        session.commit()

        feedback_count = session.query(Prediction).filter(Prediction.feedback.isnot(None)).count()

        if feedback_count % 50 == 0 and feedback_count > 0:
            background_tasks.add_task(check_drift)

        return FeedbackResponse(
            status="ok",
            feedback_received=feedback_count
        )

    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.get("/metrics")
async def metrics_endpoint():
    return Response(content=get_metrics(), media_type="text/plain")



def retrain_model(training_data: List[Dict], epochs: int = 1, batch_size: int = 4,
                  learning_rate: float = 2e-5, output_dir: str = "models/finetuned") -> Dict:
    global model

    if model is None:
        return {"success": False, "error": "Model not loaded"}

    try:
        result = model.fine_tune(
            feedback_data=training_data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_dir
        )
        return result
    except Exception as e:
        logger.error(f"Retrain error: {e}")
        return {"success": False, "error": str(e)}


def check_drift():
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post("http://localhost:8000/drift/check")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Drift check completed. Status: {data.get('status')}, Has drift: {data.get('has_drift')}")
            else:
                logger.warning(f"Drift check failed with status {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"Error during drift check: {e}")


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    session = Session()
    try:
        total = session.query(Prediction).count()
        with_feedback = session.query(Prediction).filter(Prediction.feedback.isnot(None)).count()
        correct = session.query(Prediction).filter(Prediction.feedback == True).count()
        toxic_count = session.query(Prediction).filter(Prediction.label == "toxic").count()

        active_model = session.query(ModelVersion).filter(ModelVersion.is_active == True).first()

        day_ago = datetime.now() - timedelta(days=1)
        predictions_24h = session.query(Prediction).filter(
            Prediction.timestamp >= day_ago
        ).count()

        feedback_24h = session.query(Prediction).filter(
            Prediction.feedback.isnot(None),
            Prediction.timestamp >= day_ago
        ).count()

        recent_predictions = session.query(Prediction).order_by(
            Prediction.timestamp.desc()
        ).limit(10).all()

        return StatsResponse(
            total_predictions=total,
            feedback_received=with_feedback,
            accuracy_from_feedback=correct / with_feedback if with_feedback else 0.0,
            toxic_rate=toxic_count / total if total else 0.0,
            predictions_24h=predictions_24h,
            feedback_24h=feedback_24h,
            active_model={
                "version": active_model.version,
                "f1_score": active_model.f1_score,
                "accuracy": active_model.accuracy,
                "created_at": active_model.created_at.isoformat()
            } if active_model else None,
            recent_predictions=[
                {
                    "id": p.id,
                    "text": p.text[:100] + "..." if len(p.text) > 100 else p.text,
                    "label": p.label,
                    "confidence": p.confidence,
                    "timestamp": p.timestamp.isoformat()
                }
                for p in recent_predictions
            ]
        )
    finally:
        session.close()


@app.get("/health", response_model=HealthResponse)
async def health():
    global model

    session = Session()
    try:
        total = session.query(Prediction).count()
    finally:
        session.close()

    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_path=MODEL_PATH if model else None,
        drift_detector=drift_detector is not None,
        device=model.device if model else None,
        total_predictions=total
    )


@app.post("/reload")
async def reload_model():
    global model
    if init_model():
        return {"status": "ok", "message": "Model reloaded successfully"}
    raise HTTPException(status_code=500, detail="Failed to reload model")


@app.get("/model-info")
async def model_info():
    global model
    if model:
        return model.get_model_info()
    raise HTTPException(status_code=503, detail="Model not loaded")


@app.get("/feedback-data")
async def get_feedback_data(
        limit: int = 100,
        offset: int = 0,
        has_feedback: Optional[bool] = None
):
    session = Session()
    try:
        query = session.query(Prediction)

        if has_feedback is not None:
            if has_feedback:
                query = query.filter(Prediction.feedback.isnot(None))
            else:
                query = query.filter(Prediction.feedback.is_(None))

        total = query.count()
        records = query.order_by(Prediction.timestamp.desc()).offset(offset).limit(limit).all()

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "records": [
                {
                    "id": p.id,
                    "text": p.text,
                    "cleaned_text": p.cleaned_text,
                    "toxic_prob": p.toxic_prob,
                    "label": p.label,
                    "confidence": p.confidence,
                    "feedback": p.feedback,
                    "timestamp": p.timestamp.isoformat()
                }
                for p in records
            ]
        }
    finally:
        session.close()


@app.delete("/feedback/{prediction_id}")
async def delete_feedback(prediction_id: int):
    session = Session()
    try:
        pred = session.query(Prediction).filter(Prediction.id == prediction_id).first()
        if not pred:
            raise HTTPException(status_code=404, detail="Prediction not found")

        pred.feedback = None
        session.commit()

        return {"status": "ok", "message": f"Feedback removed for prediction {prediction_id}"}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Delete feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.get("/drift-logs")
async def get_drift_logs(limit: int = 50, offset: int = 0):
    session = Session()
    try:
        logs = session.query(DriftLog).order_by(
            DriftLog.timestamp.desc()
        ).offset(offset).limit(limit).all()

        total = session.query(DriftLog).count()

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "logs": [
                {
                    "id": log.id,
                    "timestamp": log.timestamp.isoformat(),
                    "drift_score": log.drift_score,
                    "drift_detected": log.drift_detected,
                    "details": log.details
                }
                for log in logs
            ]
        }
    finally:
        session.close()


@app.get("/model-versions")
async def get_model_versions(limit: int = 20):
    session = Session()
    try:
        versions = session.query(ModelVersion).order_by(
            ModelVersion.created_at.desc()
        ).limit(limit).all()

        return {
            "versions": [
                {
                    "id": v.id,
                    "version": v.version,
                    "path": v.path,
                    "f1_score": v.f1_score,
                    "accuracy": v.accuracy,
                    "precision": v.precision,
                    "recall": v.recall,
                    "is_active": v.is_active,
                    "created_at": v.created_at.isoformat()
                }
                for v in versions
            ]
        }
    finally:
        session.close()

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "active_page": "dashboard"
        },
    )


@app.get("/inference", response_class=HTMLResponse)
async def inference_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="inference.html",
        context={
            "active_page": "inference"
        },
    )


@app.get("/experiments", response_class=HTMLResponse)
async def experiments_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="experiments.html",
        context={
            "active_page": "experiments"
        },
    )

@app.get("/api/metrics")
async def api_metrics():
    try:
        stats = await get_stats()
        drift_status_data = await get_drift_status() if drift_detector else {"status": "not_initialized"}

        session = Session()
        try:
            active_model = session.query(ModelVersion).filter(ModelVersion.is_active == True).first()
            last_drift = session.query(DriftLog).order_by(DriftLog.timestamp.desc()).first()
        finally:
            session.close()

        data_drift = 0.0
        concept_drift = 0.0
        target_drift = 0.0
        if last_drift:
            try:
                details = json.loads(last_drift.details)
                if "data_drift" in details:
                    data_drift = details["data_drift"].get("score", 0.0)
                if "concept_drift" in details and details["concept_drift"]:
                    concept_drift = details["concept_drift"].get("score", 0.0)
                if "target_drift" in details and details["target_drift"]:
                    target_drift = details["target_drift"].get("score", 0.0)
            except Exception:
                pass

        return {
            "system": {
                "total_processed": stats.total_predictions,
                "avg_latency_ms": 0,
                "uptime_hours": 0,
                "rps": 0
            },
            "quality": {
                "accuracy": active_model.accuracy if active_model else 0.0,
                "precision": active_model.precision if active_model else 0.0,
                "recall": active_model.recall if active_model else 0.0,
                "f1": active_model.f1_score if active_model else 0.0
            },
            "drift": {
                "data_drift": data_drift,
                "concept_drift": concept_drift,
                "target_drift": target_drift
            }
        }
    except Exception as e:
        logger.error(f"Error in /api/metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/latest")
async def api_predictions_latest(limit: int = 10):
    session = Session()
    try:
        records = session.query(Prediction).order_by(Prediction.timestamp.desc()).limit(limit).all()
        return [
            {
                "id": p.id,
                "text": p.text,
                "is_toxic": p.label == "toxic",
                "confidence": p.confidence or 0.0,
                "timestamp": p.timestamp.isoformat()
            }
            for p in records
        ]
    finally:
        session.close()


@app.get("/api/history")
async def api_history():
    session = Session()
    try:
        versions = session.query(ModelVersion).order_by(ModelVersion.created_at.asc()).all()
        quality_history = [
            {
                "date": v.created_at.isoformat(),
                "accuracy": v.accuracy or 0.0,
                "precision": v.precision or 0.0,
                "recall": v.recall or 0.0,
                "f1": v.f1_score or 0.0
            }
            for v in versions
        ]

        drift_logs = session.query(DriftLog).order_by(DriftLog.timestamp.asc()).all()
        drift_history = []
        for log in drift_logs:
            try:
                details = json.loads(log.details) if log.details else {}
                drift_history.append({
                    "date": log.timestamp.isoformat(),
                    "data_drift": details.get("data_drift", {}).get("score", 0.0),
                    "concept_drift": details.get("concept_drift", {}).get("score", 0.0),
                    "target_drift": details.get("target_drift", {}).get("score", 0.0)
                })
            except:
                pass

        return {
            "quality": quality_history,
            "drift": drift_history
        }
    finally:
        session.close()


@app.get("/api/experiments")
async def api_experiments():
    session = Session()
    try:
        versions = session.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
        return {
            "experiments": [
                {
                    "version": v.version,
                    "timestamp": v.created_at.isoformat(),
                    "metrics": {
                        "accuracy": v.accuracy or 0.0,
                        "f1": v.f1_score or 0.0,
                        "precision": v.precision or 0.0,
                        "recall": v.recall or 0.0
                    },
                    "status": "active" if v.is_active else "archived"
                }
                for v in versions
            ],
            "latest_version": versions[0].version if versions else None
        }
    finally:
        session.close()

@app.post("/api/retrain")
async def api_retrain(
    background_tasks: BackgroundTasks,
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 0.01,
    output_dir: str = "models/finetuned",
    days_back: int = 30
):
    session = Session()
    try:
        start_date = datetime.now() - timedelta(days=days_back)
        feedback_data = session.query(Prediction).filter(
            Prediction.feedback.isnot(None),
            Prediction.timestamp >= start_date
        ).all()

        if len(feedback_data) < 10:
            return {
                "success": False,
                "message": f"Need at least 50 feedback samples, got {len(feedback_data)}. "
                           f"Please collect more feedback before retraining."
            }

        training_data = []
        for p in feedback_data:
            is_actually_toxic = (p.label == "toxic")
            if not p.feedback:
                is_actually_toxic = not is_actually_toxic
            training_data.append({
                "text": p.text,
                "correct_label": 1 if is_actually_toxic else 0
            })
    finally:
        session.close()

    def train():
        try:
            logger.info(f"Starting retraining with {len(training_data)} samples, "
                        f"epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

            result = retrain_model(
                training_data,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                output_dir=output_dir
            )

            if result is not None:
                logger.info(f"Retraining completed. Metrics: {result}")

                session2 = Session()
                try:
                    session2.query(ModelVersion).update({ModelVersion.is_active: False})
                    session2.commit()

                    new_version = ModelVersion(
                        version=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        path=output_dir,
                        f1_score=result.get("eval_f1"),
                        accuracy=result.get("eval_accuracy"),
                        precision=result.get("eval_precision"),
                        recall=result.get("eval_recall"),
                        is_active=True
                    )
                    session2.add(new_version)
                    session2.commit()
                    logger.info(f"Saved model version: {new_version.version}")
                except Exception as e:
                    session2.rollback()
                    logger.error(f"Failed to save model version: {e}")
                finally:
                    session2.close()

                global model
                if model:
                    model.end_mlflow_run()

                model = create_model_with_mlflow(
                    use_mlflow=USE_MLFLOW,
                    experiment_name=EXPERIMENT_NAME,
                    model_path=output_dir
                )
                set_model(model)
                logger.info(f"Model reloaded from {output_dir}")
                logger.info(f"New F1: {result.get('eval_f1', 'N/A')}")
            else:
                logger.error(f"Retraining failed: {result}")
        except Exception as e:
            logger.error(f"Retraining error: {e}", exc_info=True)

    background_tasks.add_task(train)

    return {
        "success": True,
        "message": f"Retraining started with {len(training_data)} samples. "
                   f"Check server logs for progress."
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
