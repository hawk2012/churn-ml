"""Тесты инференса: корректность выходов, монотонность, устойчивость.

Тесты, требующие обученную модель, помечены `requires_model` и пропускаются,
если артефакт ещё не собран (`python -m src.evaluate`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as cfg

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

needs_model = pytest.mark.skipif(
    not cfg.MODEL_PATH.exists(),
    reason="Нет обученной модели: сначала запустите python -m src.evaluate",
)


@pytest.fixture(scope="module")
def predictor():
    from src.predict import ChurnPredictor

    return ChurnPredictor()


@needs_model
def test_predictor_loads_artifact(predictor) -> None:
    assert predictor.pipeline is not None
    assert 0.0 < predictor.threshold < 1.0
    assert len(predictor.feature_names) > 0


@needs_model
def test_probability_in_unit_range(predictor, sample_customer, loyal_customer) -> None:
    proba = predictor.predict_proba([sample_customer, loyal_customer])
    assert proba.shape == (2,)
    assert ((proba >= 0) & (proba <= 1)).all()


@needs_model
def test_risk_bands_are_ordered(predictor) -> None:
    assert predictor.risk_band(0.95) == "критический"
    assert predictor.risk_band(0.60) == "высокий"
    assert predictor.risk_band(0.40) == "средний"
    assert predictor.risk_band(0.05) == "низкий"


@needs_model
def test_high_risk_customer_scores_above_low_risk(predictor, sample_customer,
                                                  loyal_customer) -> None:
    """Содержательная проверка: клиент на месяц-в-месяц рискованнее лояльного."""
    risky, loyal = predictor.predict_proba([sample_customer, loyal_customer])
    assert risky > loyal
    assert risky > 0.5
    assert loyal < 0.3


@needs_model
def test_score_output_schema(predictor, sample_customer) -> None:
    out = predictor.score([sample_customer])
    assert list(out.columns) == [
        cfg.ID_COLUMN, "churn_probability", "churn_prediction", "risk_band"
    ]
    assert out.iloc[0]["churn_prediction"] in (0, 1)


@needs_model
def test_threshold_override_changes_labels(predictor, sample_customer) -> None:
    """Порог — управляемая ручка: при 0.01 помечаем всех, при 0.99 — почти никого."""
    low = predictor.score([sample_customer], threshold=0.01).iloc[0]["churn_prediction"]
    high = predictor.score([sample_customer], threshold=0.99).iloc[0]["churn_prediction"]
    assert low == 1
    assert high == 0


@needs_model
def test_batch_equals_single_predictions(predictor, sample_customer, loyal_customer) -> None:
    batch = predictor.predict_proba([sample_customer, loyal_customer])
    single_a = predictor.predict_proba(sample_customer)[0]
    single_b = predictor.predict_proba(loyal_customer)[0]
    assert batch[0] == pytest.approx(single_a)
    assert batch[1] == pytest.approx(single_b)


@needs_model
def test_prediction_order_is_preserved(predictor, sample_customer, loyal_customer) -> None:
    proba = predictor.predict_proba([sample_customer, loyal_customer])
    reversed_proba = predictor.predict_proba([loyal_customer, sample_customer])
    assert proba[0] == pytest.approx(reversed_proba[1])
    assert proba[1] == pytest.approx(reversed_proba[0])


@needs_model
def test_missing_optional_columns_use_defaults(predictor) -> None:
    """API подставляет значения по умолчанию — модель не должна падать."""
    minimal = {
        "customerID": "MIN-1",
        "tenure": 10,
        "MonthlyCharges": 60.0,
        "TotalCharges": 600.0,
        "Contract": "Month-to-month",
    }
    proba = predictor.predict_proba(minimal)
    assert np.isfinite(proba).all()


@needs_model
def test_zero_tenure_does_not_break_inference(predictor) -> None:
    """tenure=0 даёт деление на ноль в инженерии — проверяем обработку."""
    new_customer = {
        "customerID": "NEW-1",
        "tenure": 0,
        "MonthlyCharges": 70.0,
        "TotalCharges": 0.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
    }
    proba = predictor.predict_proba(new_customer)
    assert np.isfinite(proba).all()
    assert 0.0 <= float(proba[0]) <= 1.0


@needs_model
def test_longer_tenure_lowers_risk_on_average(predictor) -> None:
    """Ожидаемая монотонность: с ростом срока обслуживания риск падает.

    Проверяем на реальных данных: средний скор коротких контрактов выше,
    чем у клиентов с большим стажем.
    """
    from src.data import load_processed

    _, test = load_processed()
    X = test.drop(columns=[cfg.TARGET]).copy()
    proba = predictor.predict_proba(X)

    short = proba[X["tenure"] <= 6].mean()
    long = proba[X["tenure"] >= 48].mean()
    assert short > long


@needs_model
def test_metrics_are_reasonable(predictor) -> None:
    """Санити-чек качества: модель не должна деградировать ниже случайной."""
    roc_auc = predictor.metrics.get("roc_auc")
    assert roc_auc is not None
    assert roc_auc > 0.75, f"ROC-AUC упал: {roc_auc}"
    assert predictor.metrics.get("pr_auc") > 0.5
