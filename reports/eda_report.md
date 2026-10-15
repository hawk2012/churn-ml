# EDA: Telco Customer Churn

## Структура
- Строк: **7043**, столбцов: **21**
- Пустых ячеек: 0
- `TotalCharges` с пробелом вместо числа: **11** (пропуск, закодированный строкой)
- Дубликатов строк: 0, дубликатов `customerID`: 0
- Числовые: `SeniorCitizen, tenure, MonthlyCharges`
- Категориальные: 17 шт.

## Целевая переменная
- Доля оттока: **26.5%** — классы несбалансированы (соотношение ≈ 2.8:1)
- Вывод: `accuracy` бесполезна, смотрим ROC-AUC / PR-AUC / recall при фиксированной точности

## Корреляции с целевой переменной
- `tenure`: -0.352
- `TotalCharges`: -0.199
- `MonthlyCharges`: +0.193
- `SeniorCitizen`: +0.151

## Сегменты с наибольшей разницей в оттоке
- `Contract`
- `InternetService`
- `OnlineSecurity`
- `TechSupport`
- `OnlineBackup`
- `DeviceProtection`

## Графики
- `reports/figures/01_target_balance.png`
- `reports/figures/02_numeric_distributions.png`
- `reports/figures/03_churn_rate_by_category.png`
- `reports/figures/04_correlation.png`

## Гипотезы для моделирования
- Тип контракта (`Month-to-month`) — сильнейший драйвер оттока
- Короткий `tenure` связан с уходом: риск сконцентрирован в первые месяцы
- Высокий `MonthlyCharges` при отсутствии `TechSupport`/`OnlineSecurity` повышает риск
- `TotalCharges` ≈ `tenure × MonthlyCharges` — сильная корреляция, нужна аккуратная инженерия