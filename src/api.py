"""REST API для скоринга клиентов на FastAPI.

Эндпоинты:
    GET  /health          — статус сервиса и метрики модели
    GET  /model/info      — карточка модели (метрики, порог, признаки)
    POST /predict         — скоринг одного клиента
    POST /predict/batch   — пакетный скоринг (до 1000 записей)

Запуск:
    uvicorn src.api:app --host 127.0.0.1 --port 8000
    python -m src.api   (то же самое с настройками по умолчанию)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config as cfg
from src.predict import ChurnPredictor

cfg.setup_console()

MODEL: ChurnPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружает модель один раз при старте — не на каждый запрос."""
    global MODEL
    MODEL = ChurnPredictor()
    yield
    MODEL = None


app = FastAPI(
    title="Churn Scoring API",
    description="Предсказание вероятности оттока клиента телеком-оператора",
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerFeatures(BaseModel):
    """Сырые признаки клиента — те же поля, что в исходном CSV."""

    customerID: str = Field(..., description="Идентификатор клиента")
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = Field(..., ge=0, description="Срок обслуживания, месяцев")
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }]
        }
    }


class PredictionResponse(BaseModel):
    customerID: str
    churn_probability: float
    churn_prediction: int
    risk_band: str
    threshold: float
    model_roc_auc: float | None = None


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]
    summary: dict[str, Any]


def _require_model() -> ChurnPredictor:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return MODEL


@app.get("/health", summary="Статус сервиса")
def health() -> dict:
    model = _require_model()
    return {
        "status": "ok",
        "model_loaded": True,
        "threshold": model.threshold,
        "roc_auc": model.metrics.get("roc_auc"),
    }


@app.get("/model/info", summary="Карточка модели")
def model_info() -> dict:
    model = _require_model()
    return {
        "model_type": model.metrics.get("model", "unknown"),
        "threshold": model.threshold,
        "metrics": {
            k: v for k, v in model.metrics.items()
            if k in ("roc_auc", "pr_auc", "precision", "recall", "f1", "brier",
                     "accuracy", "test_rows")
        },
        "n_features": len(model.feature_names),
        "risk_bands": {
            "критический": ">= 0.70",
            "высокий": "0.50 – 0.70",
            "средний": "0.30 – 0.50",
            "низкий": "< 0.30",
        },
    }


@app.post("/predict", response_model=PredictionResponse, summary="Скоринг клиента")
def predict(customer: CustomerFeatures) -> PredictionResponse:
    model = _require_model()
    result = model.score(customer.model_dump()).iloc[0]
    return PredictionResponse(
        customerID=result[cfg.ID_COLUMN],
        churn_probability=float(result["churn_probability"]),
        churn_prediction=int(result["churn_prediction"]),
        risk_band=result["risk_band"],
        threshold=model.threshold,
        model_roc_auc=model.metrics.get("roc_auc"),
    )


@app.post("/predict/batch", response_model=BatchResponse, summary="Пакетный скоринг")
def predict_batch(request: BatchRequest) -> BatchResponse:
    model = _require_model()
    rows = [c.model_dump() for c in request.customers]
    scored = model.score(rows)

    predictions = [
        PredictionResponse(
            customerID=r[cfg.ID_COLUMN],
            churn_probability=float(r["churn_probability"]),
            churn_prediction=int(r["churn_prediction"]),
            risk_band=r["risk_band"],
            threshold=model.threshold,
        )
        for _, r in scored.iterrows()
    ]

    band_counts = scored["risk_band"].value_counts().to_dict()
    return BatchResponse(
        count=len(predictions),
        predictions=predictions,
        summary={
            "flagged_for_retention": int(scored["churn_prediction"].sum()),
            "flagged_share": round(float(scored["churn_prediction"].mean()), 4),
            "mean_probability": round(float(scored["churn_probability"].mean()), 4),
            "risk_bands": band_counts,
        },
    )


def main() -> None:
    import uvicorn

    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
