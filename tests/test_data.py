"""Тесты загрузки, очистки и разбиения данных."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.data import clean, encode_target, load_raw, split


def test_load_raw_has_expected_shape(raw_df: pd.DataFrame) -> None:
    assert raw_df.shape[0] == 7043
    assert cfg.TARGET in raw_df.columns
    assert cfg.ID_COLUMN in raw_df.columns


def test_totalcharges_is_string_in_raw_data(raw_df: pd.DataFrame) -> None:
    """Пропуски в TotalCharges закодированы пробелом — источник бага, если не заметить."""
    assert raw_df["TotalCharges"].dtype.kind == "O"
    blanks = (raw_df["TotalCharges"].astype(str).str.strip() == "").sum()
    assert blanks == 11


def test_clean_converts_totalcharges_to_numeric(raw_df: pd.DataFrame) -> None:
    cleaned = clean(raw_df)
    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_clean_sets_zero_totalcharges_for_zero_tenure(raw_df: pd.DataFrame) -> None:
    """Клиент с tenure=0 ещё не платил — логично 0, а не медиана."""
    cleaned = clean(raw_df)
    zero_tenure = cleaned[cleaned["tenure"] == 0]
    assert len(zero_tenure) > 0
    assert (zero_tenure["TotalCharges"] == 0).all()


def test_clean_is_idempotent(raw_df: pd.DataFrame) -> None:
    once = clean(raw_df)
    twice = clean(once)
    pd.testing.assert_frame_equal(once, twice)


def test_encode_target_maps_yes_no(raw_df: pd.DataFrame) -> None:
    encoded = encode_target(clean(raw_df))
    assert set(encoded[cfg.TARGET].unique()) == {0, 1}
    assert encoded[cfg.TARGET].dtype.kind == "i"


def test_encode_target_positive_rate_matches_source(raw_df: pd.DataFrame) -> None:
    encoded = encode_target(clean(raw_df))
    expected = (raw_df[cfg.TARGET] == "Yes").mean()
    assert encoded[cfg.TARGET].mean() == pytest.approx(expected, abs=1e-9)


def test_split_preserves_target_ratio(cleaned_df: pd.DataFrame) -> None:
    train, test = split(cleaned_df)
    overall = cleaned_df[cfg.TARGET].mean()
    assert train[cfg.TARGET].mean() == pytest.approx(overall, abs=0.005)
    assert test[cfg.TARGET].mean() == pytest.approx(overall, abs=0.01)


def test_split_has_no_overlap(cleaned_df: pd.DataFrame) -> None:
    train, test = split(cleaned_df)
    assert set(train[cfg.ID_COLUMN]).isdisjoint(set(test[cfg.ID_COLUMN]))


def test_split_is_reproducible(cleaned_df: pd.DataFrame) -> None:
    train_a, test_a = split(cleaned_df)
    train_b, test_b = split(cleaned_df)
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)


def test_split_sizes_match_config(cleaned_df: pd.DataFrame) -> None:
    train, test = split(cleaned_df)
    assert len(train) + len(test) == len(cleaned_df)
    assert len(test) == pytest.approx(len(cleaned_df) * cfg.TEST_SIZE, abs=2)


def test_no_missing_values_after_clean(cleaned_df: pd.DataFrame) -> None:
    assert cleaned_df.isna().sum().sum() == 0
