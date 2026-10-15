"""Единая конфигурация проекта: пути, константы, гиперпараметры по умолчанию."""
from __future__ import annotations

import sys
from pathlib import Path


def setup_console() -> None:
    """Windows-консоль по умолчанию в cp1251 — кириллица в логах ломается."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

# --- Пути ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw" / "telco_churn.csv"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "churn_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

for _p in (DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --- Данные ---
TARGET = "Churn"
ID_COLUMN = "customerID"

# --- Воспроизводимость ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Метрики / бизнес ---
# Цена ошибок для подбора порога: пропустить уходящего клиента дороже,
# чем зря потревожить лояльного (стоимость удержания << потеря LTV).
COST_FALSE_NEGATIVE = 10.0  # не удержали -> потеряли клиента
COST_FALSE_POSITIVE = 1.0   # удержали зря -> скидка/звонок
