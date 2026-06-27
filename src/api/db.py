from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text)
    cleaned_text = Column(Text, nullable=True)
    toxic_prob = Column(Float)
    label = Column(String(10))
    confidence = Column(Float, nullable=True)
    threshold_used = Column(Float, default=0.5)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    feedback = Column(Boolean, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "cleaned_text": self.cleaned_text,
            "toxic_prob": self.toxic_prob,
            "label": self.label,
            "confidence": self.confidence,
            "threshold_used": self.threshold_used,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "feedback": self.feedback
        }


class ModelVersion(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(50))
    path = Column(String(500))
    f1_score = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "version": self.version,
            "path": self.path,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class DriftLog(Base):
    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    drift_score = Column(Float)
    drift_detected = Column(Boolean)
    details = Column(Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "drift_score": self.drift_score,
            "drift_detected": self.drift_detected,
            "details": self.details
        }


engine = create_engine(
    "sqlite:///toxicity.db",
    connect_args={"check_same_thread": False}
)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)