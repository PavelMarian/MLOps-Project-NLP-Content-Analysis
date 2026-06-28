import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir / "src" / "api"))

from src.api.api import app
from src.api.db import Base, Session, Prediction, ReferenceData


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_prediction(db_session):
    pred = Prediction(
        text="тестовое сообщение",
        cleaned_text="тестовое сообщение",
        toxic_prob=0.8,
        label="toxic",
        confidence=0.8,
        threshold_used=0.5,
        feedback=None,
    )
    db_session.add(pred)
    db_session.commit()
    db_session.refresh(pred)
    return pred