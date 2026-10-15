"""Обучение и сравнение моделей с кросс-валидацией.

Сравниваем 4 семейства моделей на одинаковом пайплайне признаков:
логистическая регрессия (baseline), случайный лес, XGBoost, LightGBM.
Метрики — ROC-AUC и PR-AUC: классы несбалансированы (отток ~26.5%),
accuracy здесь обманывает.

Запуск: python -m src.train [--tune]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from src import config as cfg
from src.data import load_processed
from src.features import build_pipeline

cfg.setup_console()


def candidate_models(scale_pos_weight: float) -> dict:
    """Набор моделей-кандидатов. Дисбаланс учитываем через веса классов."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=cfg.RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=cfg.RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=cfg.RANDOM_STATE,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=cfg.RANDOM_STATE,
            verbose=-1,
        ),
    }


def compare(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    """Кросс-валидация всех кандидатов с одинаковыми фолдами."""
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    rows = []
    for name, estimator in candidate_models(scale_pos_weight).items():
        pipe = build_pipeline(estimator)
        t0 = time.perf_counter()
        res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
        elapsed = time.perf_counter() - t0

        row = {"model": name, "fit_seconds": round(elapsed, 1)}
        for metric in scoring:
            scores = res[f"test_{metric}"]
            row[f"{metric}_mean"] = round(float(scores.mean()), 4)
            row[f"{metric}_std"] = round(float(scores.std()), 4)
        rows.append(row)
        print(f"{name:20s} ROC-AUC={row['roc_auc_mean']:.4f} "
              f"(±{row['roc_auc_std']:.4f})  PR-AUC={row['pr_auc_mean']:.4f}  "
              f"F1={row['f1_mean']:.4f}  [{elapsed:.1f}s]")

    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение моделей оттока")
    parser.add_argument("--tune", action="store_true",
                        help="подобрать гиперпараметры лучшей модели через Optuna")
    parser.add_argument("--n-trials", type=int, default=40,
                        help="число испытаний Optuna (по умолчанию 40)")
    args = parser.parse_args()

    train, _ = load_processed()
    X = train.drop(columns=[cfg.TARGET])
    y = train[cfg.TARGET]

    print(f"train: {X.shape}, доля положительного класса: {y.mean():.3f}\n")
    results = compare(X, y)
    results.to_csv(cfg.REPORTS_DIR / "model_comparison.csv", index=False)

    print("\nИтоговая таблица (по ROC-AUC):")
    print(results[["model", "roc_auc_mean", "pr_auc_mean", "f1_mean",
                   "recall_mean", "precision_mean", "fit_seconds"]].to_string(index=False))

    best_name = results.iloc[0]["model"]
    print(f"\nЛучшая модель: {best_name}")

    if args.tune:
        from src.tune import tune_model

        best_params = tune_model(best_name, X, y, n_trials=args.n_trials)
        (cfg.REPORTS_DIR / "best_params.json").write_text(
            json.dumps({"model": best_name, "params": best_params},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Параметры сохранены: {cfg.REPORTS_DIR / 'best_params.json'}")

    (cfg.REPORTS_DIR / "model_comparison.md").write_text(
        "# Сравнение моделей (5-fold CV)\n\n"
        + results.to_markdown(index=False)
        + f"\n\n**Лучшая модель:** {best_name}\n",
        encoding="utf-8",
    )
    print(f"Отчёт: {cfg.REPORTS_DIR / 'model_comparison.md'}")


if __name__ == "__main__":
    main()
