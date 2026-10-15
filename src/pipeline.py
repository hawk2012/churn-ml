"""Полный пайплайн: данные -> признаки -> обучение -> оценка -> объяснение.

Одна команда для воспроизведения всего результата с нуля.

Запуск: python -m src.pipeline
"""
from __future__ import annotations

import time

from src import config as cfg
from src.data import load_processed
from src.data import main as prepare_data
from src.evaluate import main as evaluate
from src.explain import main as explain
from src.train import compare

cfg.setup_console()


def step(name: str, func) -> float:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    t0 = time.perf_counter()
    func()
    elapsed = time.perf_counter() - t0
    print(f"-> {name}: {elapsed:.1f} c")
    return elapsed


def compare_models() -> None:
    """Сравнение моделей без разбора CLI-аргументов (их читает только src.train)."""
    train_df, _ = load_processed()
    X = train_df.drop(columns=[cfg.TARGET])
    y = train_df[cfg.TARGET]

    results = compare(X, y)
    results.to_csv(cfg.REPORTS_DIR / "model_comparison.csv", index=False)
    (cfg.REPORTS_DIR / "model_comparison.md").write_text(
        "# Сравнение моделей (5-fold CV)\n\n" + results.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    print("Пайплайн предсказания оттока клиентов")
    total = 0.0
    total += step("1/4 Подготовка данных", prepare_data)
    total += step("2/4 Сравнение моделей (5-fold CV)", compare_models)
    total += step("3/4 Обучение и оценка финальной модели", evaluate)
    total += step("4/4 SHAP-интерпретация", explain)

    print(f"\n{'=' * 60}")
    print(f"Готово за {total:.1f} c")
    print(f"Модель:  {cfg.MODEL_PATH}")
    print(f"Метрики: {cfg.METRICS_PATH}")
    print(f"Графики: {cfg.FIGURES_DIR}")
    print("Запуск API: uvicorn src.api:app --port 8000")



if __name__ == "__main__":
    main()
