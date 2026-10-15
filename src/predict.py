"""Инференс: загрузка сохранённой модели и скоринг новых клиентов.

Модель хранится вместе с препроцессингом, поэтому на вход подаются сырые
строки в том же виде, что и в исходном CSV — никакой ручной подготовки.

Запуск:
    python -m src.predict --customer-id 7590-VHVEG
    python -m src.predict --top 20
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src import config as cfg
from src.data import load_processed

cfg.setup_console()

RISK_BANDS = [
    (0.70, "критический"),
    (0.50, "высокий"),
    (0.30, "средний"),
    (0.00, "низкий"),
]


class ChurnPredictor:
    """Обёртка над сохранённым артефактом: вероятности, метки, риск-бэнды."""

    def __init__(self, model_path=cfg.MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}. Сначала запустите python -m src.evaluate"
            )
        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.threshold = float(artifact["threshold"])
        self.metrics = artifact.get("metrics", {})
        self.feature_names = artifact.get("feature_names", [])

    @staticmethod
    def _prepare(rows: pd.DataFrame | dict | list[dict]) -> pd.DataFrame:
        if isinstance(rows, dict):
            df = pd.DataFrame([rows])
        elif isinstance(rows, list):
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame(rows)
        return df.drop(columns=[cfg.TARGET], errors="ignore")

    @staticmethod
    def risk_band(probability: float) -> str:
        for bound, label in RISK_BANDS:
            if probability >= bound:
                return label
        return "низкий"

    def predict_proba(self, rows) -> np.ndarray:
        """Вероятность оттока для одной или нескольких записей."""
        X = self._prepare(rows)
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, rows, threshold: float | None = None) -> np.ndarray:
        """Бинарная метка с учётом рабочего порога."""
        thr = self.threshold if threshold is None else threshold
        return (self.predict_proba(rows) >= thr).astype(int)

    def score(self, rows, threshold: float | None = None) -> pd.DataFrame:
        """Полный результат скоринга: id, вероятность, метка, риск-бэнд."""
        X = self._prepare(rows)
        proba = self.predict_proba(X)
        thr = self.threshold if threshold is None else threshold

        out = pd.DataFrame({
            "churn_probability": np.round(proba, 4),
            "churn_prediction": (proba >= thr).astype(int),
            "risk_band": [self.risk_band(p) for p in proba],
        })
        if cfg.ID_COLUMN in X.columns:
            out.insert(0, cfg.ID_COLUMN, X[cfg.ID_COLUMN].astype(str).values)
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Скоринг оттока клиентов")
    parser.add_argument("--customer-id", help="ID клиента из тестовой выборки")
    parser.add_argument("--top", type=int, help="показать N клиентов с наибольшим риском")
    parser.add_argument("--threshold", type=float, help="переопределить порог")
    parser.add_argument("--json", action="store_true", help="вывод в JSON")
    args = parser.parse_args()

    predictor = ChurnPredictor()
    _, test = load_processed()

    print(f"Модель: ROC-AUC={predictor.metrics.get('roc_auc')} | "
          f"порог={predictor.threshold:.3f}\n")

    if args.customer_id:
        row = test[test[cfg.ID_COLUMN] == args.customer_id]
        if row.empty:
            print(f"Клиент {args.customer_id} не найден в тестовой выборке")
            return
        result = predictor.score(row, args.threshold).iloc[0]
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"Клиент: {result[cfg.ID_COLUMN]}")
            print(f"Вероятность оттока: {result['churn_probability']:.3f}")
            print(f"Решение: {'уйдёт' if result['churn_prediction'] else 'останется'} "
                  f"| риск: {result['risk_band']}")
        return

    n = args.top or 10
    scored = predictor.score(test, args.threshold).sort_values(
        "churn_probability", ascending=False
    )
    print(f"Топ-{n} клиентов по риску оттока:\n")
    print(scored.head(n).to_string(index=False))

    print(f"\nРаспределение по риск-бэндам:")
    print(scored["risk_band"].value_counts().to_string())


if __name__ == "__main__":
    main()
