from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from collections import deque
import logging
import time
import sys
from pathlib import Path
from db import Session, Prediction, ModelVersion, DriftLog

sys.path.append(str(Path(__file__).parent.parent))

from models.model import create_model_with_mlflow
from drift_monitoring.drift_detector import DriftDetector, DriftAnalyzer, metrics

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import atexit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Toxicity Detection MLOps API",
    description="API для определения токсичности текстов с MLOps функционалом",
    version="1.0.0"
)

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
async def predict(request: PredictRequest):
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


@app.get("/drift", response_model=DriftResponse)
async def get_drift():
    session = Session()
    try:
        last_drift = session.query(DriftLog).order_by(DriftLog.timestamp.desc()).first()

        if last_drift:
            return DriftResponse(
                drift_detected=last_drift.drift_detected,
                drift_score=last_drift.drift_score,
                last_check=last_drift.timestamp.isoformat()
            )
        return DriftResponse(
            drift_detected=False,
            drift_score=0.0,
            last_check=None
        )
    finally:
        session.close()


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(request: RetrainRequest, background_tasks: BackgroundTasks):
    session = Session()
    try:
        start_date = datetime.now() - timedelta(days=request.days_back)
        feedback_data = session.query(Prediction).filter(
            Prediction.feedback.isnot(None),
            Prediction.timestamp >= start_date
        ).all()

        if len(feedback_data) < 50:
            return RetrainResponse(
                success=False,
                message=f"Need at least 50 feedback samples, got {len(feedback_data)}"
            )

        training_data = []
        for p in feedback_data:
            is_actually_toxic = (p.label == "toxic")
            if not p.feedback:
                is_actually_toxic = not is_actually_toxic

            training_data.append({
                "text": p.text,
                "correct_label": 1 if is_actually_toxic else 0
            })

        def train():
            result = retrain_model(
                training_data,
                request.epochs,
                request.batch_size,
                request.learning_rate,
                request.output_dir
            )

            if result and result.get("success"):
                session2 = Session()
                try:
                    session2.query(ModelVersion).update({ModelVersion.is_active: False})

                    new_version = ModelVersion(
                        version=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        path=result.get("model_path", request.output_dir),
                        f1_score=result.get("eval_f1"),
                        accuracy=result.get("eval_accuracy"),
                        precision=result.get("eval_precision"),
                        recall=result.get("eval_recall"),
                        is_active=True
                    )
                    session2.add(new_version)
                    session2.commit()

                    logger.info(f"Model retrained! F1: {result.get('eval_f1', 0):.3f}")

                    global model
                    if model:
                        try:
                            model.end_mlflow_run()
                        except:
                            pass

                    model = create_model_with_mlflow(
                        use_mlflow=USE_MLFLOW,
                        experiment_name=EXPERIMENT_NAME
                    )
                    logger.info("Model reloaded after retraining")

                except Exception as e:
                    session2.rollback()
                    logger.error(f"Error saving model metadata: {e}")
                finally:
                    session2.close()

        background_tasks.add_task(train)

        return RetrainResponse(
            success=True,
            message=f"Retraining started with {len(training_data)} samples"
        )

    except Exception as e:
        logger.error(f"Retrain error: {e}")
        return RetrainResponse(success=False, message=str(e))
    finally:
        session.close()


def retrain_model(training_data: List[Dict], epochs: int = 3, batch_size: int = 16,
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
    global drift_detector

    session = Session()
    try:
        recent = session.query(Prediction).filter(
            Prediction.timestamp >= datetime.now() - timedelta(days=1)
        ).all()

        if len(recent) < 50:
            return

        current_texts = [p.text for p in recent]

        if drift_detector is None:
            init_drift_detector()

        if drift_detector:
            result = drift_detector.detect(current_texts)

            log = DriftLog(
                drift_score=result["drift_score"],
                drift_detected=result["drift_detected"],
                details=str(result)
            )
            session.add(log)
            session.commit()

            if result["drift_detected"]:
                logger.warning(f"Data drift detected! Score: {result['drift_score']:.3f}")

            return result
        return None

    except Exception as e:
        logger.error(f"Drift check error: {e}")
        return None
    finally:
        session.close()


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )