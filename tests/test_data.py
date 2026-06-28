import io
import pandas as pd
from fastapi.testclient import TestClient
from src.api.load_new_data import normalize_label
from src.api.db import Session, Prediction


def test_normalize_label():
    assert normalize_label(1) == "toxic"
    assert normalize_label(0) == "non-toxic"
    assert normalize_label("toxic") == "toxic"
    assert normalize_label("not toxic") == "non-toxic"
    assert normalize_label("Toxic") == "toxic"
    assert normalize_label("Not Toxic") == "non-toxic"
    assert normalize_label("positive") == "toxic"
    assert normalize_label("negative") == "non-toxic"
    assert normalize_label("") is None
    assert normalize_label("unknown") is None


def test_load_data_endpoint(client, db_session, monkeypatch):
    csv_data = "text,label\n'это хорошо',not toxic\n'ты идиот',toxic"
    files = {"file": ("test.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
    data = {"clear_existing": "false", "text_column": "text", "label_column": "label"}

    response = client.post("/data/load-new-data", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    assert result["inserted"] == 2
    assert result["clear_existing"] is False

    session = Session()
    count = session.query(Prediction).filter(Prediction.feedback == True).count()
    assert count >= 2
    session.close()


def test_load_data_invalid_labels(client):
    csv_data = "text,label\n'это хорошо',maybe\n'ты идиот',toxic"
    files = {"file": ("test.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
    response = client.post("/data/load-new-data", files=files, data={"clear_existing": "false"})
    assert response.status_code == 400
    assert "неподдерживаемые метки" in response.json()["detail"]


def test_load_data_missing_columns(client):
    csv_data = "col1,col2\n1,2"
    files = {"file": ("test.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
    response = client.post("/data/load-new-data", files=files)
    assert response.status_code == 400
    assert "Колонка 'text' не найдена" in response.json()["detail"]