"""Интерпретация модели: SHAP-значения и объяснение отдельных клиентов.

Показывает, какие признаки двигают прогноз и в какую сторону — это то,
что уходит в бизнес: не «модель сказала 0.73», а «риск из-за месячного
контракта и отсутствия техподдержки».

Запуск: python -m src.explain
Результат: графики SHAP в reports/figures/, таблица важности в reports/
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config as cfg
from src.data import load_processed

cfg.setup_console()


def load_artifact() -> dict:
    if not cfg.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {cfg.MODEL_PATH}. Сначала запустите python -m src.evaluate"
        )
    return joblib.load(cfg.MODEL_PATH)


def transformed_matrix(artifact: dict, X: pd.DataFrame) -> np.ndarray:
    """Прогоняет сырые данные через препроцессинг пайплайна (без модели)."""
    pre = artifact["pipeline"][:-1]  # engineer -> columns
    return pre.transform(X)


def explain(artifact: dict, X: pd.DataFrame, sample: int = 2000) -> pd.DataFrame:
    """Считает SHAP-значения для выборки и возвращает важность признаков."""
    X_sample = X.sample(min(sample, len(X)), random_state=cfg.RANDOM_STATE)
    Xt = transformed_matrix(artifact, X_sample)
    names = artifact["feature_names"]

    model = artifact["pipeline"].named_steps["model"]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(Xt)

    mean_abs = np.abs(values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return importance, values, Xt, names


def plot_importance(importance: pd.DataFrame, top_n: int = 20) -> None:
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["mean_abs_shap"], color="steelblue")
    ax.set_xlabel("Средний |SHAP| (влияние на логит)")
    ax.set_title(f"Топ-{top_n} признаков по важности (SHAP)")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "10_shap_importance.png", dpi=130)
    plt.close(fig)


def plot_beeswarm(values: np.ndarray, Xt: np.ndarray, names: list[str]) -> None:
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(values, Xt, feature_names=names, max_display=15, show=False)
    plt.title("SHAP: направление влияния признаков", fontsize=11)
    plt.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "11_shap_beeswarm.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def explain_customer(artifact: dict, row: pd.Series, top_k: int = 6) -> dict:
    """Объяснение одного клиента: топ-признаки, сдвинувшие прогноз."""
    X = pd.DataFrame([row]).drop(columns=[cfg.TARGET], errors="ignore")
    Xt = transformed_matrix(artifact, X)
    names = artifact["feature_names"]
    model = artifact["pipeline"].named_steps["model"]

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(Xt)[0]
    base = float(explainer.expected_value)
    proba = float(model.predict_proba(Xt)[0, 1])

    contrib = (
        pd.DataFrame({
            "feature": names,
            "value": np.asarray(Xt[0], dtype=float),
            "shap": values,
        })
        .assign(abs_shap=lambda d: d["shap"].abs())
        .sort_values("abs_shap", ascending=False)
        .head(top_k)
        .drop(columns="abs_shap")
    )

    return {
        "customerID": str(row.get(cfg.ID_COLUMN, "unknown")),
        "churn_probability": round(proba, 4),
        "base_value_logit": round(base, 4),
        "top_factors": contrib.to_dict(orient="records"),
    }


def main() -> None:
    artifact = load_artifact()
    _, test = load_processed()
    X_test = test.drop(columns=[cfg.TARGET])

    importance, values, Xt, names = explain(artifact, X_test)
    plot_importance(importance)
    plot_beeswarm(values, Xt, list(names))

    importance.to_csv(cfg.REPORTS_DIR / "shap_importance.csv", index=False)

    # Примеры объяснений: самый рискованный и самый «спокойный» клиент
    proba = artifact["pipeline"].predict_proba(X_test)[:, 1]
    examples = {
        "highest_risk": explain_customer(artifact, test.iloc[int(np.argmax(proba))]),
        "lowest_risk": explain_customer(artifact, test.iloc[int(np.argmin(proba))]),
    }
    (cfg.REPORTS_DIR / "shap_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Топ-10 признаков по SHAP:")
    for _, r in importance.head(10).iterrows():
        print(f"  {r['feature']:28s} {r['mean_abs_shap']:.4f}")

    print(f"\nПример (макс. риск): p={examples['highest_risk']['churn_probability']:.3f}")
    for f in examples["highest_risk"]["top_factors"][:3]:
        print(f"  {f['feature']:28s} shap={f['shap']:+.3f}")

    print(f"\nГрафики: {cfg.FIGURES_DIR}")
    print(f"Важность: {cfg.REPORTS_DIR / 'shap_importance.csv'}")
    print(f"Примеры: {cfg.REPORTS_DIR / 'shap_examples.json'}")


if __name__ == "__main__":
    main()
