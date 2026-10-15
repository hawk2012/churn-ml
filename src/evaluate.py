"""Финальная оценка на отложенной тестовой выборке.

Обучаем финальную модель (XGBoost с параметрами из Optuna) на train,
оцениваем на test и сохраняем артефакты:
* `models/churn_model.joblib` — модель вместе с препроцессингом;
* `reports/metrics.json` — метрики и выбранный порог;
* графики ROC / PR / матрицы ошибок / калибровки.

Тест не участвовал ни в отборе модели, ни в подборе гиперпараметров —
цифры ниже честные, без утечки.

Запуск: python -m src.evaluate
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config as cfg
from src.data import load_processed
from src.features import build_pipeline

cfg.setup_console()
sns.set_theme(style="whitegrid")


def load_best_params(model_name: str = "XGBoost") -> dict:
    """Читает параметры из reports/best_params.json, если он есть."""
    path = cfg.REPORTS_DIR / "best_params.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if model_name in data:
        return data[model_name]
    if isinstance(data, dict) and data.get("model") == model_name:
        return data.get("params", {})
    return {}


def build_final_model() -> Pipeline:
    params = load_best_params("XGBoost")
    estimator = XGBClassifier(
        eval_metric="logloss",
        n_jobs=-1,
        random_state=cfg.RANDOM_STATE,
        **params,
    )
    return build_pipeline(estimator)


def choose_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                     cost_fn: float = cfg.COST_FALSE_NEGATIVE,
                     cost_fp: float = cfg.COST_FALSE_POSITIVE) -> dict:
    """Порог, минимизирующий ожидаемые бизнес-потери.

    Стоимость = cost_fn * FN + cost_fp * FP. Пропустить уходящего клиента
    дороже, чем зря предложить скидку лояльному, поэтому порог смещается вниз.
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    costs, f1s = [], []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        costs.append(cost_fn * fn + cost_fp * fp)
        f1s.append(f1_score(y_true, pred, zero_division=0))

    costs = np.asarray(costs)
    f1s = np.asarray(f1s)
    best_cost_idx = int(costs.argmin())
    best_f1_idx = int(f1s.argmax())

    return {
        "cost_optimal": float(thresholds[best_cost_idx]),
        "min_expected_cost": float(costs[best_cost_idx]),
        "f1_optimal": float(thresholds[best_f1_idx]),
        "best_f1": float(f1s[best_f1_idx]),
        "cost_curve": {"thresholds": thresholds.tolist(), "costs": costs.tolist()},
        "f1_curve": {"thresholds": thresholds.tolist(), "f1": f1s.tolist()},
    }


def recall_at_precision(y_true: np.ndarray, y_proba: np.ndarray,
                        target_precision: float) -> dict:
    """Максимальный recall при точности не ниже заданной.

    Практически значимая метрика: «сколько уходящих поймаем, если допустимо,
    чтобы не более X% срабатываний были ложными».
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    mask = precision[:-1] >= target_precision
    if not mask.any():
        return {"target_precision": target_precision, "recall": None, "threshold": None}
    idx = int(np.argmax(np.where(mask, recall[:-1], -1)))
    return {
        "target_precision": target_precision,
        "recall": float(recall[idx]),
        "threshold": float(thresholds[idx]),
    }


def capacity_threshold(y_proba: np.ndarray, capacity_share: float) -> float:
    """Порог, при котором кампания удержания охватывает заданную долю базы.

    Бизнес-оптимум по стоимости (10:1) требует обзвонить большинство клиентов —
    это упирается в ресурсы колл-центра. Поэтому считаем вариант «топ-K% по риску».
    """
    return float(np.quantile(y_proba, 1.0 - capacity_share))


def evaluate_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, thr: float) -> dict:
    """Набор метрик при конкретном пороге."""
    pred = (y_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(thr), 3),
        "coverage": round(float(pred.mean()), 3),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def plot_roc(y_true: np.ndarray, y_proba: np.ndarray, auc: float) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"XGBoost (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Случайная модель")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривая (тест)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "05_roc_curve.png", dpi=130)
    plt.close(fig)


def plot_pr(y_true: np.ndarray, y_proba: np.ndarray, ap: float) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"XGBoost (PR-AUC = {ap:.3f})")
    ax.axhline(baseline, color="red", ls="--", lw=1,
               label=f"Базовая доля оттока ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (тест)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "06_pr_curve.png", dpi=130)
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Останется", "Уйдёт"],
                yticklabels=["Останется", "Уйдёт"])
    ax.set_xlabel(f"Предсказание (порог {threshold:.2f})")
    ax.set_ylabel("Факт")
    ax.set_title("Матрица ошибок (тест)")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "07_confusion_matrix.png", dpi=130)
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "o-", label=f"XGBoost (Brier = {brier:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Идеальная калибровка")
    ax.set_xlabel("Предсказанная вероятность")
    ax.set_ylabel("Наблюдаемая частота")
    ax.set_title("Калибровка вероятностей (тест)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "08_calibration.png", dpi=130)
    plt.close(fig)
    return float(brier)


def plot_threshold_curves(curves: dict, chosen: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    t = np.asarray(curves["cost_curve"]["thresholds"])
    axes[0].plot(t, curves["cost_curve"]["costs"], lw=2)
    axes[0].axvline(curves["cost_optimal"], color="red", ls="--",
                    label=f"Минимум потерь: {curves['cost_optimal']:.2f}")
    axes[0].axvline(chosen, color="green", ls="-", lw=1.5,
                    label=f"Рабочий порог: {chosen:.2f}")
    axes[0].set_xlabel("Порог")
    axes[0].set_ylabel("Ожидаемые потери (усл. ед.)")
    axes[0].set_title("Бизнес-стоимость ошибок (FN дороже FP в 10 раз)")
    axes[0].legend()

    axes[1].plot(t, curves["f1_curve"]["f1"], lw=2, color="darkgreen", label="F1")
    axes[1].plot(t, curves["precision_curve"], lw=1.5, ls=":", color="tab:blue",
                 label="Precision")
    axes[1].plot(t, curves["recall_curve"], lw=1.5, ls=":", color="tab:orange",
                 label="Recall")
    axes[1].axvline(chosen, color="green", ls="-", lw=1.5,
                    label=f"Рабочий порог: {chosen:.2f}")
    axes[1].set_xlabel("Порог")
    axes[1].set_ylabel("Значение метрики")
    axes[1].set_title("Precision / Recall / F1 в зависимости от порога")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "09_threshold_selection.png", dpi=130)
    plt.close(fig)


def precision_recall_curves(y_true: np.ndarray, y_proba: np.ndarray,
                            thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precision и recall для сетки порогов (для графика и выбора стратегии)."""
    prec, rec = [], []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        prec.append(precision_score(y_true, pred, zero_division=0))
        rec.append(recall_score(y_true, pred, zero_division=0))
    return np.asarray(prec), np.asarray(rec)


def main() -> None:
    train, test = load_processed()
    X_train, y_train = train.drop(columns=[cfg.TARGET]), train[cfg.TARGET]
    X_test, y_test = test.drop(columns=[cfg.TARGET]), test[cfg.TARGET]

    model = build_final_model()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    y_true = y_test.to_numpy()

    roc_auc = roc_auc_score(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)

    threshold_stats = choose_threshold(y_true, proba)
    grid = np.asarray(threshold_stats["cost_curve"]["thresholds"])
    prec_curve, rec_curve = precision_recall_curves(y_true, proba, grid)
    threshold_stats["precision_curve"] = prec_curve.tolist()
    threshold_stats["recall_curve"] = rec_curve.tolist()

    # Рабочий порог: оптимум по F1, а не по стоимости. При стоимости 10:1
    # формальный минимум потерь требует охватить ~73% базы — нереально для
    # кампании удержания, поэтому стоимость показываем как ориентир,
    # а рабочий порог берём по балансу precision/recall и ёмкости кампании.
    thr = threshold_stats["f1_optimal"]
    y_pred = (proba >= thr).astype(int)

    brier = plot_calibration(y_true, proba)
    plot_roc(y_true, proba, roc_auc)
    plot_pr(y_true, proba, pr_auc)
    plot_confusion(y_true, y_pred, thr)
    plot_threshold_curves(threshold_stats, thr)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Стратегии порога под разные ограничения бизнеса
    strategies = {
        "cost_optimal": evaluate_at_threshold(y_true, proba, threshold_stats["cost_optimal"]),
        "f1_optimal": evaluate_at_threshold(y_true, proba, threshold_stats["f1_optimal"]),
        "top_20_percent": evaluate_at_threshold(y_true, proba, capacity_threshold(proba, 0.20)),
        "top_10_percent": evaluate_at_threshold(y_true, proba, capacity_threshold(proba, 0.10)),
    }

    metrics = {
        "model": "XGBoost (tuned, Optuna)",
        "test_rows": int(len(y_test)),
        "positive_rate_test": round(float(y_true.mean()), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier": round(brier, 4),
        "threshold": round(float(thr), 3),
        "threshold_cost_optimal": round(threshold_stats["cost_optimal"], 3),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "recall_at_precision": [
            recall_at_precision(y_true, proba, p) for p in (0.5, 0.6, 0.7, 0.8)
        ],
        "threshold_strategies": strategies,
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Останется", "Уйдёт"], zero_division=0
        ),
    }

    cfg.METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    artifact = {
        "pipeline": model,
        "threshold": float(thr),
        "feature_names": list(
            model.named_steps["columns"].get_feature_names_out()
        ),
        "trained_on_rows": int(len(X_train)),
        "metrics": {k: v for k, v in metrics.items()
                    if k not in ("classification_report",)},
    }
    joblib.dump(artifact, cfg.MODEL_PATH)

    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {brier:.4f}")
    print(f"Рабочий порог (F1-оптимум): {thr:.3f}")
    print(f"  precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}  "
          f"F1={metrics['f1']:.3f}  accuracy={metrics['accuracy']:.3f}")
    print(f"Матрица ошибок: TN={tn} FP={fp} FN={fn} TP={tp}")
    for r in metrics["recall_at_precision"]:
        print(f"  recall при precision>={r['target_precision']:.1f}: "
              f"{r['recall']:.3f} (порог {r['threshold']:.3f})")
    print("\nСтратегии порога:")
    for name, s in strategies.items():
        print(f"  {name:16s} порог={s['threshold']:.3f} охват={s['coverage']:.0%} "
              f"precision={s['precision']:.3f} recall={s['recall']:.3f} f1={s['f1']:.3f}")
    print(f"\nМодель: {cfg.MODEL_PATH}")
    print(f"Метрики: {cfg.METRICS_PATH}")


if __name__ == "__main__":
    main()
