from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import json
from sqlalchemy import func

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from drift_monitoring.drift_detector import DriftDetector, DriftReport
from db import Session, Prediction, DriftLog, ReferenceData
from drift_monitoring.prometheus_client import metrics

router = APIRouter(prefix="/drift", tags=["drift"])

drift_detector = None
REFERENCE_SIZE = 500

def init_drift_detector():
    global drift_detector
    session = Session()
    try:
        reference = session.query(ReferenceData).order_by(func.random()).limit(REFERENCE_SIZE).all()
        if len(reference) > 100:
            texts = [r.text for r in reference]
            labels = [r.label for r in reference]
            drift_detector = DriftDetector(
                reference_texts=texts,
                reference_labels=labels
            )
            print(f"Drift detector initialized with {len(texts)} reference samples")
        else:
            print(f"Not enough feedback data for drift detection (need >100, got {len(recent)})")
    finally:
        session.close()

def get_drift_detector():
    global drift_detector
    if drift_detector is None:
        init_drift_detector()
    return drift_detector

@router.post("/check")
async def check_drift(background_tasks: BackgroundTasks):
    detector = get_drift_detector()
    if detector is None:
        raise HTTPException(status_code=400, detail="Drift detector not initialized")

    session = Session()
    try:
        day_ago = datetime.now() - timedelta(days=1)
        current = session.query(Prediction).filter(
            Prediction.timestamp >= day_ago
        ).all()

        if len(current) < 10:
            return {
                "status": "skipped",
                "message": f"Not enough current data (need 50, got {len(current)})"
            }

        current_texts = [p.text for p in current]
        current_labels = [1 if p.label == "toxic" else 0 for p in current]

        report = detector.generate_report(
            current_texts=current_texts,
            y_reference=detector.reference_labels,
            y_current=current_labels,
            log_to_mlflow=True
        )

        drift_log = DriftLog(
            drift_score=report.data_drift.score if report.data_drift else 0,
            drift_detected=report.has_drift(),
            details=str(report.to_dict())
        )
        session.add(drift_log)
        session.commit()

        metrics.update_drift_metrics(report.to_dict())

        try:
            base_dir = Path(__file__).parent.parent.parent
            reports_dir = base_dir / "reports"
            reports_dir.mkdir(exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drift_report_{timestamp_str}.json"
            filepath = reports_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            print(f"Drift report saved to {filepath}")
        except Exception as e:
            print(f"Failed to save drift report: {e}")

        return {
            "status": "ok",
            "report": report.to_dict(),
            "has_drift": report.has_drift()
        }
    finally:
        session.close()


@router.get("/status")
async def get_drift_status():
    detector = get_drift_detector()
    if detector is None:
        return {"status": "not_initialized"}

    summary = detector.get_drift_summary()

    session = Session()
    try:
        last_log = session.query(DriftLog).order_by(
            DriftLog.timestamp.desc()
        ).first()

        if last_log:
            summary["last_db_entry"] = {
                "timestamp": last_log.timestamp.isoformat(),
                "drift_score": last_log.drift_score,
                "drift_detected": last_log.drift_detected
            }
    finally:
        session.close()

    return summary


@router.get("/history")
async def get_drift_history(limit: int = 100):
    session = Session()
    try:
        logs = session.query(DriftLog).order_by(
            DriftLog.timestamp.desc()
        ).limit(limit).all()

        return [
            {
                "timestamp": log.timestamp.isoformat(),
                "drift_score": log.drift_score,
                "drift_detected": log.drift_detected,
                "details": log.details
            }
            for log in logs
        ]
    finally:
        session.close()


@router.post("/reset")
async def reset_drift_detector():
    global drift_detector
    drift_detector = None
    init_drift_detector()

    detector = get_drift_detector()
    if detector:
        return {"status": "ok", "message": "Drift detector reinitialized"}
    else:
        return {"status": "error", "message": "Failed to reinitialize drift detector"}