"""Общие фикстуры для тестов."""
from __future__ import annotations

import pandas as pd
import pytest

from src import config as cfg


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Сырой датасет — основа для тестов очистки."""
    return pd.read_csv(cfg.DATA_RAW)


@pytest.fixture(scope="session")
def cleaned_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    from src.data import clean, encode_target

    return encode_target(clean(raw_df))


@pytest.fixture(scope="session")
def sample_customer() -> dict:
    """Один клиент с высоким риском оттока."""
    return {
        "customerID": "TEST-RISK",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5,
        "TotalCharges": 95.5,
    }


@pytest.fixture(scope="session")
def loyal_customer() -> dict:
    """Клиент с низким риском: долгий срок и двухлетний контракт."""
    return {
        "customerID": "TEST-LOYAL",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 68,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 45.0,
        "TotalCharges": 3060.0,
    }
