"""Подбор гиперпараметров через Optuna.

Оптимизируем ROC-AUC на стратифицированной кросс-валидации.
Важно: целевая метрика считается внутри CV на train, тест не участвует —
иначе подбор превратится в подгонку под тестовую выборку.

Запуск: python -m src.tune [--model XGBoost] [--trials 60]
"""
from __future__ import annotations

import argparse
import json
import warnings

import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src import config as cfg
from src.data import load_processed
from src.features import build_pipeline

cfg.setup_console()
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest(trial: optuna.Trial, name: str, scale_pos_weight: float):
    """Пространство поиска для каждой модели."""
    if name == "LogisticRegression":
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 50.0, log=True),
            penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
            solver="liblinear",
            class_weight="balanced",
            max_iter=3000,
            random_state=cfg.RANDOM_STATE,
        )

    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 300, 1200, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=cfg.RANDOM_STATE,
        )

    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-4, 5.0, log=True),
            scale_pos_weight=trial.suggest_float(
                "scale_pos_weight", 0.5 * scale_pos_weight, 2.0 * scale_pos_weight
            ),
            eval_metric="logloss",
            n_jobs=-1,
            random_state=cfg.RANDOM_STATE,
        )

    if name == "LightGBM":
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            scale_pos_weight=trial.suggest_float(
                "scale_pos_weight", 0.5 * scale_pos_weight, 2.0 * scale_pos_weight
            ),
            n_jobs=-1,
            random_state=cfg.RANDOM_STATE,
            verbose=-1,
        )

    raise ValueError(f"Неизвестная модель: {name}")


def tune_model(name: str, X: pd.DataFrame, y: pd.Series, n_trials: int = 40,
               n_splits: int = 5) -> dict:
    """Возвращает лучшие гиперпараметры для указанной модели."""
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        estimator = _suggest(trial, name, scale_pos_weight)
        pipe = build_pipeline(estimator)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"[{name}] лучший ROC-AUC на CV: {study.best_value:.4f} "
          f"(испытаний: {len(study.trials)})")
    print(f"[{name}] параметры: {study.best_params}")

    study.trials_dataframe().to_csv(
        cfg.REPORTS_DIR / f"optuna_trials_{name}.csv", index=False
    )
    return study.best_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Подбор гиперпараметров через Optuna")
    parser.add_argument("--model", default=None, help="имя модели (по умолчанию — все)")
    parser.add_argument("--trials", type=int, default=60, help="число испытаний")
    args = parser.parse_args()

    train, _ = load_processed()
    X = train.drop(columns=[cfg.TARGET])
    y = train[cfg.TARGET]

    models = [args.model] if args.model else [
        "LogisticRegression", "XGBoost", "LightGBM", "RandomForest"
    ]

    results = {}
    for name in models:
        results[name] = tune_model(name, X, y, n_trials=args.trials)

    out = cfg.REPORTS_DIR / "best_params.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
