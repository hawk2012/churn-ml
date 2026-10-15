"""Разведочный анализ данных: структура, пропуски, дисбаланс, зависимости.

Запуск: python -m src.eda
Результат: графики в reports/figures/ и текстовый отчёт reports/eda_report.md
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-режим: без окна, только файлы

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config as cfg

cfg.setup_console()
sns.set_theme(style="whitegrid", palette="deep")


def load_raw() -> pd.DataFrame:
    """Читает сырой CSV. TotalCharges приходит строкой (есть пробелы) — оставляем как есть."""
    df = pd.read_csv(cfg.DATA_RAW)
    return df


def profile(df: pd.DataFrame) -> dict:
    """Базовая сводка: размеры, пропуски, дубликаты, кардинальность."""
    n_missing = int(df.isna().sum().sum())
    # В сыром файле пропуски TotalCharges закодированы пробелом, а не NaN
    blank_total = int((df["TotalCharges"].astype(str).str.strip() == "").sum())
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_cells": n_missing,
        "totalcharges_blank": blank_total,
        "duplicated_rows": int(df.duplicated().sum()),
        "duplicated_ids": int(df[cfg.ID_COLUMN].duplicated().sum()),
        "numeric_cols": df.select_dtypes(include=np.number).columns.tolist(),
        "categorical_cols": [c for c in df.select_dtypes(include=["object", "string"]).columns
                             if c != "TotalCharges"],
    }


def plot_target_balance(df: pd.DataFrame) -> None:
    counts = df[cfg.TARGET].value_counts()
    share = df[cfg.TARGET].value_counts(normalize=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=axes[0], hue=counts.index, legend=False)
    axes[0].set_title("Отток: абсолютные значения")
    axes[0].set_ylabel("Клиентов")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v}\n({share.iloc[i]:.1f}%)", ha="center", va="bottom")

    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90,
                colors=sns.color_palette("deep")[:2])
    axes[1].set_title("Доля классов")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "01_target_balance.png", dpi=130)
    plt.close(fig)


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    d = df.copy()
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce")
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, col in enumerate(num_cols):
        sns.histplot(data=d, x=col, hue=cfg.TARGET, kde=True, ax=axes[0, j],
                     element="step", common_norm=False)
        axes[0, j].set_title(f"Распределение: {col}")
        sns.boxplot(data=d, x=cfg.TARGET, y=col, ax=axes[1, j], hue=cfg.TARGET, legend=False)
        axes[1, j].set_title(f"Boxplot: {col}")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "02_numeric_distributions.png", dpi=130)
    plt.close(fig)


def plot_categorical_churn_rate(df: pd.DataFrame, top_n: int = 6) -> None:
    """Доля оттока по категориям — ищем сегменты с повышенным риском."""
    # TotalCharges читается строкой из-за пробелов, но по смыслу он числовой —
    # как категорию его не рассматриваем, иначе получаем график из 6500 столбцов.
    cat_cols = [c for c in df.select_dtypes(include=["object", "string"]).columns
                if c not in (cfg.ID_COLUMN, cfg.TARGET, "TotalCharges")]

    # Оставляем признаки с наибольшей разницей в доле оттока
    scored = []
    for col in cat_cols:
        rate = df.groupby(col, observed=True)[cfg.TARGET].apply(lambda s: (s == "Yes").mean())
        scored.append((col, float(rate.max() - rate.min())))
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [c for c, _ in scored[:top_n]]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.ravel(), selected):
        rate = (df.groupby(col, observed=True)[cfg.TARGET]
                  .apply(lambda s: (s == "Yes").mean())
                  .sort_values(ascending=False))
        sns.barplot(x=rate.values, y=rate.index, ax=ax, hue=rate.index, legend=False)
        ax.axvline(df[cfg.TARGET].eq("Yes").mean(), color="red", ls="--", lw=1,
                   label="Средний отток")
        ax.set_title(f"Доля оттока по «{col}»")
        ax.set_xlabel("Доля оттока")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "03_churn_rate_by_category.png", dpi=130)
    plt.close(fig)
    return selected


def plot_correlation(df: pd.DataFrame) -> None:
    d = df.copy()
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce")
    d[cfg.TARGET] = d[cfg.TARGET].map({"Yes": 1, "No": 0})
    num = d[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", cfg.TARGET]]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(num.corr(numeric_only=True), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, square=True)
    ax.set_title("Корреляции числовых признаков")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "04_correlation.png", dpi=130)
    plt.close(fig)
    return num.corr(numeric_only=True)[cfg.TARGET].drop(cfg.TARGET).to_dict()


def main() -> None:
    df = load_raw()
    info = profile(df)
    corr = plot_correlation(df)
    top_cats = plot_categorical_churn_rate(df)
    plot_target_balance(df)
    plot_numeric_distributions(df)

    churn_rate = df[cfg.TARGET].eq("Yes").mean()
    lines = [
        "# EDA: Telco Customer Churn",
        "",
        "## Структура",
        f"- Строк: **{info['rows']}**, столбцов: **{info['cols']}**",
        f"- Пустых ячеек: {info['missing_cells']}",
        f"- `TotalCharges` с пробелом вместо числа: **{info['totalcharges_blank']}** "
        "(пропуск, закодированный строкой)",
        f"- Дубликатов строк: {info['duplicated_rows']}, дубликатов `customerID`: {info['duplicated_ids']}",
        f"- Числовые: `{', '.join(info['numeric_cols'])}`",
        f"- Категориальные: {len(info['categorical_cols'])} шт.",
        "",
        "## Целевая переменная",
        f"- Доля оттока: **{churn_rate:.1%}** — классы несбалансированы "
        f"(соотношение ≈ {(1 - churn_rate) / churn_rate:.1f}:1)",
        "- Вывод: `accuracy` бесполезна, смотрим ROC-AUC / PR-AUC / recall при фиксированной точности",
        "",
        "## Корреляции с целевой переменной",
        *[f"- `{k}`: {v:+.3f}" for k, v in sorted(corr.items(), key=lambda x: -abs(x[1]))],
        "",
        "## Сегменты с наибольшей разницей в оттоке",
        *[f"- `{c}`" for c in top_cats],
        "",
        "## Графики",
        "- `reports/figures/01_target_balance.png`",
        "- `reports/figures/02_numeric_distributions.png`",
        "- `reports/figures/03_churn_rate_by_category.png`",
        "- `reports/figures/04_correlation.png`",
        "",
        "## Гипотезы для моделирования",
        "- Тип контракта (`Month-to-month`) — сильнейший драйвер оттока",
        "- Короткий `tenure` связан с уходом: риск сконцентрирован в первые месяцы",
        "- Высокий `MonthlyCharges` при отсутствии `TechSupport`/`OnlineSecurity` повышает риск",
        "- `TotalCharges` ≈ `tenure × MonthlyCharges` — сильная корреляция, нужна аккуратная инженерия",
    ]
    (cfg.REPORTS_DIR / "eda_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"EDA готово. Строк: {info['rows']}, доля оттока: {churn_rate:.1%}")
    print(f"Отчёт: {cfg.REPORTS_DIR / 'eda_report.md'}")
    print(f"Графики: {cfg.FIGURES_DIR}")


if __name__ == "__main__":
    main()
