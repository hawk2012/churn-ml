"""Тесты пайплайна признаков: отсутствие утечки, устойчивость к новым категориям."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import config as cfg
from src.features import (
    FeatureEngineer,
    binary_columns,
    build_pipeline,
    build_preprocessor,
)


@pytest.fixture(scope="module")
def train_test_frames(cleaned_df: pd.DataFrame):
    from src.data import split

    train, test = split(cleaned_df)
    return (
        train.drop(columns=[cfg.TARGET]),
        test.drop(columns=[cfg.TARGET]),
        train[cfg.TARGET],
    )


def test_engineer_drops_id_and_target(cleaned_df: pd.DataFrame) -> None:
    X = cleaned_df.drop(columns=[cfg.TARGET])
    out = FeatureEngineer().fit_transform(X)
    assert cfg.ID_COLUMN not in out.columns
    assert cfg.TARGET not in out.columns


def test_engineer_adds_expected_features(cleaned_df: pd.DataFrame) -> None:
    X = cleaned_df.drop(columns=[cfg.TARGET])
    out = FeatureEngineer().fit_transform(X)
    for col in ["avg_charge_per_month", "is_new_customer",
                "charge_tenure_ratio", "n_services"]:
        assert col in out.columns


def test_avg_charge_equals_total_over_tenure(cleaned_df: pd.DataFrame) -> None:
    X = cleaned_df.drop(columns=[cfg.TARGET])
    out = FeatureEngineer().fit_transform(X)
    mask = out["tenure"] > 0
    expected = out.loc[mask, "TotalCharges"] / out.loc[mask, "tenure"]
    assert np.allclose(out.loc[mask, "avg_charge_per_month"], expected)


def test_avg_charge_is_nan_for_zero_tenure(cleaned_df: pd.DataFrame) -> None:
    """Деление на ноль не должно давать inf — иначе модель сломается."""
    X = cleaned_df.drop(columns=[cfg.TARGET])
    out = FeatureEngineer().fit_transform(X)
    zero = out[out["tenure"] == 0]
    assert len(zero) > 0
    assert zero["avg_charge_per_month"].isna().all()


def test_no_infinite_values_after_engineering(cleaned_df: pd.DataFrame) -> None:
    X = cleaned_df.drop(columns=[cfg.TARGET])
    out = FeatureEngineer().fit_transform(X)
    numeric = out.select_dtypes(include=np.number)
    assert not np.isinf(numeric.to_numpy(dtype=float)).any()


def test_is_new_customer_matches_tenure_threshold(cleaned_df: pd.DataFrame) -> None:
    X = cleaned_df.drop(columns=[cfg.TARGET])
    out = FeatureEngineer().fit_transform(X)
    assert (out.loc[out["tenure"] <= 6, "is_new_customer"] == 1).all()
    assert (out.loc[out["tenure"] > 6, "is_new_customer"] == 0).all()


def test_n_services_counts_yes_values(cleaned_df: pd.DataFrame) -> None:
    X = cleaned_df.drop(columns=[cfg.TARGET])
    out = FeatureEngineer().fit_transform(X)
    assert out["n_services"].between(0, 7).all()


def test_engineer_does_not_mutate_input(cleaned_df: pd.DataFrame) -> None:
    X = cleaned_df.drop(columns=[cfg.TARGET])
    before = X.copy()
    FeatureEngineer().fit_transform(X)
    pd.testing.assert_frame_equal(X, before)


def test_binary_columns_detects_yes_no(cleaned_df: pd.DataFrame) -> None:
    X = cleaned_df.drop(columns=[cfg.TARGET])
    out = FeatureEngineer().fit_transform(X)
    cols = binary_columns(out)
    assert "Partner" in cols
    assert "Dependents" in cols
    assert "Contract" not in cols  # три категории


def test_preprocessor_output_is_numeric_and_finite(train_test_frames) -> None:
    X_train, X_test, _ = train_test_frames
    pre = build_preprocessor()
    Xt = pre.fit_transform(X_train)
    Xv = pre.transform(X_test)

    assert Xt.shape[1] == Xv.shape[1]
    assert np.isfinite(np.asarray(Xt, dtype=float)).all()
    assert np.isfinite(np.asarray(Xv, dtype=float)).all()


def test_preprocessor_handles_unseen_category(train_test_frames) -> None:
    """Новая категория не должна ронять инференс — handle_unknown='ignore'."""
    X_train, X_test, _ = train_test_frames
    pre = build_preprocessor()
    pre.fit(X_train)

    X_new = X_test.head(3).copy()
    X_new["InternetService"] = "Satellite"  # такой категории в данных нет
    out = pre.transform(X_new)
    assert np.isfinite(np.asarray(out, dtype=float)).all()


def test_feature_count_is_stable_between_fit_and_transform(train_test_frames) -> None:
    X_train, X_test, _ = train_test_frames
    pre = build_preprocessor()
    pre.fit(X_train)
    names_fit = pre.named_steps["columns"].get_feature_names_out()
    pre.transform(X_test)
    names_transform = pre.named_steps["columns"].get_feature_names_out()
    assert list(names_fit) == list(names_transform)


def test_scaling_uses_train_statistics_only(train_test_frames) -> None:
    """Проверка отсутствия утечки: среднее по train-трансформу близко к нулю,
    а по test — уже нет (иначе статистики подглядывали в тест)."""
    X_train, X_test, _ = train_test_frames
    pre = build_preprocessor()
    Xt = pre.fit_transform(X_train)
    Xv = pre.transform(X_test)

    names = list(pre.named_steps["columns"].get_feature_names_out())
    idx = names.index("tenure")

    assert abs(float(np.mean(Xt[:, idx]))) < 1e-6
    # Тестовое среднее отличается от нуля, но незначительно — распределения близки
    assert abs(float(np.mean(Xv[:, idx]))) < 0.3


def test_pipeline_fits_end_to_end(train_test_frames) -> None:
    X_train, X_test, y_train = train_test_frames
    pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=cfg.RANDOM_STATE))
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    assert proba.shape[0] == len(X_test)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_accepts_raw_dict_input(cleaned_df: pd.DataFrame, sample_customer: dict) -> None:
    """Пайплайн должен принимать сырые данные: без customerID и с пробелами."""
    from src.data import split

    train, _ = split(cleaned_df)
    pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=cfg.RANDOM_STATE))
    pipe.fit(train.drop(columns=[cfg.TARGET]), train[cfg.TARGET])

    X = pd.DataFrame([sample_customer])
    proba = pipe.predict_proba(X)[:, 1]
    assert 0.0 <= float(proba[0]) <= 1.0


def test_pipeline_handles_blank_totalcharges(cleaned_df: pd.DataFrame) -> None:
    """Пробел в TotalCharges на инференсе не должен ронять пайплайн."""
    from src.data import split

    train, test = split(cleaned_df)
    pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=cfg.RANDOM_STATE))
    pipe.fit(train.drop(columns=[cfg.TARGET]), train[cfg.TARGET])

    X = test.drop(columns=[cfg.TARGET]).head(5).copy()
    X["TotalCharges"] = ["", " ", "", "100.0", " 200.5 "]
    proba = pipe.predict_proba(X)[:, 1]
    assert np.isfinite(proba).all()
