# Прогноз изменения веса
## Описание задачи
Проект посвящён прогнозированию изменения веса пользователя на основе данных о его питании, физической активности, качестве сна и уровне стресса. Задача — регрессия: предсказать числовое изменение веса за определённый период.
Дополнительно проведён расширенный анализ на большем датасете (UCI) для проверки закономерностей, связанных с влиянием физической активности на вес, на большей выборке.

## Датасеты
### Основной датасет — регрессия
**Comprehensive Weight Change Prediction** (Kaggle)
https://www.kaggle.com/datasets/abdullah0a/comprehensive-weight-change-prediction

100 записей. Признаки:
- Возраст, пол
- Текущий вес (lbs/kg)
- BMR (базовый обмен веществ)
- Суточная калорийность рациона
- Суточный калорийный дефицит/избыток
- Длительность периода наблюдения (недели)
- Уровень физической активности (Sedentary/Lightly Active/Moderately Active/Very Active)
- Качество сна (Poor/Fair/Good/Excellent)
- Уровень стресса
- **Целевая переменная:** изменение веса (lbs)

### Дополнительный датасет — только для анализа
**Estimation of Obesity Levels Based on Eating Habits and Physical Condition** (UCI ML Repository)
https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

2087 записей (после удаления дубликатов). Данные о пищевых привычках и физической активности жителей Мексики, Перу и Колумбии.

Используется **только для дополнительного EDA-анализа** — проверки закономерностей на большей выборке (например, связь активности и категории веса). Не участвует в обучении итоговой модели, так как решает другую задачу (классификация категории ожирения, а не прогноз числового изменения веса).

## Ограничение
Основной датасет для модели небольшой (100 записей) — результаты стоит рассматривать как предварительные. Для повышения надёжности при обучении моделей использовалась кросс-валидация вместо простого train/test split.

## Структура репозитория
```
weight-change-prediction/
├── data/
│   ├── raw/ # исходные датасеты
│   └── processed/ # данные после предобработки
├── notebooks/
│   ├── 01_eda.ipynb  # разведочный анализ данных
│   ├── 02_preprocessing.ipynb # предобработка и feature engineering
│   └── 03_modeling.ipynb # обучение и сравнение моделей
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   └── train.py
├── README.md
└── requirements.txt
```

## Использованные признаки (после feature engineering)
- Вес в кг (переведён из lbs)
- Изменение веса в неделю (нормализация по длительности периода)
- Доля калорийного дефицита/избытка относительно BMR
- Флаг выбросов (по методу IQR, не удалялись из данных)
- Закодированные категориальные признаки: пол, уровень активности, качество сна

## Модели и метрики
дописать


## Выводы


## Авторы и вклад
- Якунина Екатерина — поиск и анализ данных, EDA, предобработка, feature engineering
- Дубинина Арина — обучение моделей, подбор гиперпараметров, оценка метрик

## Источники и материалы
- Kaggle Learn: Pandas — https://www.kaggle.com/learn/pandas
- Kaggle Learn: Data Visualization — https://www.kaggle.com/learn/data-visualization
- Seaborn tutorial — https://seaborn.pydata.org/tutorial.html
- scikit-learn preprocessing — https://scikit-learn.org/stable/modules/preprocessing.html
- Примеры решений на Kaggle по основному датасету — https://www.kaggle.com/datasets/abdullah0a/comprehensive-weight-change-prediction/code