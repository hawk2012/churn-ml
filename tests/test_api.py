"""Тесты API: схемы, валидация, эндпоинты.

Используем TestClient из FastAPI — реальный сервер поднимать не нужно.
Тесты пропускаются, если модель ещё не обучена.
"""
from __future__ import annotations

import pytest

from src import config as cfg

needs_model = pytest.mark.skipif(
    not cfg.MODEL_PATH.exists(),
    reason="Нет обученной модели: сначала запустите python -m src.evaluate",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from src.api import app

    with TestClient(app) as c:
        yield c


@needs_model
def test_health_endpoint(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


@needs_model
def test_model_info_endpoint(client) -> None:
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_features"] > 0
    assert 0 < body["threshold"] < 1


@needs_model
def test_predict_high_risk_customer(client, sample_customer) -> None:
    resp = client.post("/predict", json=sample_customer)
    assert resp.status_code == 200
    body = resp.json()
    assert body["customerID"] == sample_customer["customerID"]
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_band"] in {"критический", "высокий", "средний", "низкий"}


@needs_model
def test_predict_low_risk_customer(client, loyal_customer) -> None:
    resp = client.post("/predict", json=loyal_customer)
    assert resp.status_code == 200
    assert resp.json()["churn_prediction"] == 0


@needs_model
def test_predict_requires_core_fields(client) -> None:
    """Без обязательных полей — 422, а не 500."""
    resp = client.post("/predict", json={"customerID": "X"})
    assert resp.status_code == 422


@needs_model
def test_predict_rejects_negative_tenure(client) -> None:
    payload = {
        "customerID": "NEG-1", "tenure": -1,
        "MonthlyCharges": 50.0, "TotalCharges": 100.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


@needs_model
def test_predict_rejects_negative_charges(client) -> None:
    payload = {
        "customerID": "NEG-2", "tenure": 5,
        "MonthlyCharges": -10.0, "TotalCharges": 100.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


@needs_model
def test_batch_endpoint_returns_summary(client, sample_customer, loyal_customer) -> None:
    resp = client.post("/predict/batch", json={"customers": [sample_customer, loyal_customer]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert "flagged_for_retention" in body["summary"]


@needs_model
def test_batch_rejects_empty_list(client) -> None:
    resp = client.post("/predict/batch", json={"customers": []})
    assert resp.status_code == 422


@needs_model
def test_batch_preserves_order(client, sample_customer, loyal_customer) -> None:
    resp = client.post(
        "/predict/batch", json={"customers": [loyal_customer, sample_customer]}
    )
    ids = [p["customerID"] for p in resp.json()["predictions"]]
    assert ids == [loyal_customer["customerID"], sample_customer["customerID"]]
