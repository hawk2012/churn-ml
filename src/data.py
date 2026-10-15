"""Загрузка, очистка и разбиение данных.

Ключевые решения:
* `TotalCharges` приходит строкой, потому что 11 значений записаны пробелом —
  это пропуск, закодированный текстом, а не реальный ноль.
* Разбиение train/test фиксировано (стратифицировано по целевой переменной)
  и сохраняется на диск, чтобы все последующие шаги работали с одними и теми же
  данными и результаты были воспроизводимы.

Запуск: python -m src.data
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config as cfg

cfg.setup_console()


def load_raw() -> pd.DataFrame:
    """Читает сырой CSV."""
    return pd.read_csv(cfg.DATA_RAW)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит типы и обрабатывает закодированные пропуски.

    `TotalCharges` -> float, пробелы -> NaN.
    Для клиентов с `tenure == 0` суммарные начисления логично равны 0
    (они ещё не платили), остальные пропуски заполняем медианой.
    """
    out = df.copy()

    out["TotalCharges"] = pd.to_numeric(
        out["TotalCharges"].astype(str).str.strip().replace({"": None}),
        errors="coerce",
    )

    zero_tenure = out["tenure"].eq(0) & out["TotalCharges"].isna()
    out.loc[zero_tenure, "TotalCharges"] = 0.0
    if out["TotalCharges"].isna().any():
        out["TotalCharges"] = out["TotalCharges"].fillna(out["TotalCharges"].median())

    out[cfg.ID_COLUMN] = out[cfg.ID_COLUMN].astype(str)
    return out


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Churn: Yes/No -> 1/0. Целевая переменная — положительный класс «ушёл»."""
    out = df.copy()
    out[cfg.TARGET] = out[cfg.TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return out


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Стратифицированное разбиение: сохраняем долю оттока в обеих выборках."""
    train, test = train_test_split(
        df,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=df[cfg.TARGET],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    raw = load_raw()
    df = encode_target(clean(raw))
    train, test = split(df)

    train.to_csv(cfg.DATA_PROCESSED / "train.csv", index=False)
    test.to_csv(cfg.DATA_PROCESSED / "test.csv", index=False)

    print(f"Всего: {len(df)} | train: {len(train)} | test: {len(test)}")
    print(f"Доля оттока: всего {df[cfg.TARGET].mean():.3f} | "
          f"train {train[cfg.TARGET].mean():.3f} | test {test[cfg.TARGET].mean():.3f}")
    print(f"Сохранено в {cfg.DATA_PROCESSED}")


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Читает уже подготовленные выборки; при отсутствии — пересоздаёт их."""
    train_path = cfg.DATA_PROCESSED / "train.csv"
    test_path = cfg.DATA_PROCESSED / "test.csv"
    if not (train_path.exists() and test_path.exists()):
        main()
    return pd.read_csv(train_path), pd.read_csv(test_path)


if __name__ == "__main__":
    main()
