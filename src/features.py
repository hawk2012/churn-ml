"""Пайплайн признаков.

Производные признаки считаются внутри трансформера, а не заранее, чтобы
статистики (медианы, one-hot словарь) учились только на train — иначе утечка.

Состав пайплайна:
1. `FeatureEngineer` — приводит типы, добавляет производные признаки;
2. числовые: медианная импутация + масштабирование;
3. бинарные Yes/No -> 0/1;
4. категориальные: OneHotEncoder (неизвестные категории игнорируются).

Запуск: python -m src.features
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as cfg

cfg.setup_console()

NUMERIC_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "avg_charge_per_month", "charge_tenure_ratio", "n_services",
    "is_new_customer", "SeniorCitizen",
]

SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "PhoneService",
]

# Поля, которых может не быть в запросе к API: добиваем значениями по умолчанию,
# иначе ColumnTransformer упадёт на «columns are missing» ещё до модели.
DEFAULT_CATEGORICAL_COLS = {
    "gender": "Female",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}

DEFAULT_NUMERIC_COLS = {"SeniorCitizen": 0}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Приводит типы и добавляет производные признаки, не заглядывая в test."""

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(X).copy()

        if cfg.ID_COLUMN in df.columns:
            df = df.drop(columns=[cfg.ID_COLUMN])
        if cfg.TARGET in df.columns:
            df = df.drop(columns=[cfg.TARGET])

        # Инференс может прийти без части необязательных полей: добиваем
        # их значениями по умолчанию, иначе ColumnTransformer упадёт на
        # «columns are missing» ещё до модели.
        for col, default in DEFAULT_CATEGORICAL_COLS.items():
            if col not in df.columns:
                df[col] = default
        for col, default in DEFAULT_NUMERIC_COLS.items():
            if col not in df.columns:
                df[col] = default

        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].astype(str).str.strip().replace({"": None}),
            errors="coerce",
        )
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

        # --- Инженерия признаков ---
        # Средний чек за месяц жизни клиента: сглаживает связь TotalCharges ~ tenure.
        df["avg_charge_per_month"] = df["TotalCharges"] / df["tenure"].replace(0, np.nan)
        # Новый клиент — самый рискованный сегмент по данным EDA.
        df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
        # Дорогой тариф при коротком сроке — усиленный сигнал риска.
        df["charge_tenure_ratio"] = df["MonthlyCharges"] / (df["tenure"] + 1)
        # Число подключённых услуг: чем больше «привязок», тем ниже отток.
        present = [c for c in SERVICE_COLS if c in df.columns]
        df["n_services"] = (
            sum(df[c].astype(str).eq("Yes").astype(int) for c in present)
            if present else 0
        )

        return df


def binary_columns(df: pd.DataFrame) -> list[str]:
    """Колонки, где ровно два уникальных значения вида Yes/No."""
    cols = []
    for col in df.columns:
        if col in (cfg.TARGET, cfg.ID_COLUMN):
            continue
        values = set(df[col].dropna().astype(str).unique())
        if values and values <= {"Yes", "No"}:
            cols.append(col)
    return cols


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Yes/No -> 1/0. Список колонок определяется на train."""

    def fit(self, X: pd.DataFrame, y=None) -> "BinaryEncoder":
        self.columns_ = binary_columns(pd.DataFrame(X))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(X)
        out = pd.DataFrame(index=df.index)
        for col in self.columns_:
            if col in df.columns:
                out[col] = df[col].astype(str).map({"Yes": 1, "No": 0}).astype(float)
            else:
                out[col] = 0.0
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.columns_, dtype=object)


def _make_preprocessor(numeric: list[str], binary: list[str], categorical: list[str]):
    """Собирает ColumnTransformer под конкретный набор колонок."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric,
            ),
            ("bin", BinaryEncoder(), binary),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class ColumnRouter(BaseEstimator, TransformerMixin):
    """Определяет типы колонок и строит под них препроцессор.

    Нужен, потому что состав колонок известен только после `FeatureEngineer`,
    а `ColumnTransformer` требует списки колонок при конструировании.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "ColumnRouter":
        df = pd.DataFrame(X)
        binary = binary_columns(df)
        categorical = [
            c for c in df.columns
            if c not in binary and df[c].dtype.kind not in "ifb"
        ]
        numeric = [c for c in df.columns if c not in binary and c not in categorical]

        self.preprocessor_ = _make_preprocessor(numeric, binary, categorical)
        self.preprocessor_.fit(df, y)
        self.feature_names_out_ = np.asarray(
            self.preprocessor_.get_feature_names_out(), dtype=object
        )
        return self

    def transform(self, X: pd.DataFrame):
        return self.preprocessor_.transform(pd.DataFrame(X))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_out_


def build_pipeline(estimator) -> Pipeline:
    """Полный пайплайн: инженерия -> препроцессинг -> модель.

    Всё в одном объекте: значит, `fit` на train не подглядывает в test,
    а модель можно сериализовать и подавать на вход сырые строки.
    """
    return Pipeline([
        ("engineer", FeatureEngineer()),
        ("columns", ColumnRouter()),
        ("model", clone(estimator)),
    ])


def build_preprocessor() -> Pipeline:
    """Только препроцессинг — для проверок и отладки."""
    return Pipeline([
        ("engineer", FeatureEngineer()),
        ("columns", ColumnRouter()),
    ])


def main() -> None:
    from src.data import load_processed

    train, test = load_processed()
    X_train = train.drop(columns=[cfg.TARGET])
    X_test = test.drop(columns=[cfg.TARGET])

    pre = build_preprocessor()
    Xt = pre.fit_transform(X_train)
    Xv = pre.transform(X_test)

    names = pre.named_steps["columns"].get_feature_names_out()
    print(f"Признаков на выходе: {len(names)}")
    print(f"train: {Xt.shape} | test: {Xv.shape}")
    print(f"Примеры: {', '.join(map(str, names[:12]))} ...")


if __name__ == "__main__":
    main()