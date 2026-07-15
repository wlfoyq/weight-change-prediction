# -*- coding: utf-8 -*-
"""
Feature engineering для основного датасета (Weight Change Prediction).
Логика 1:1 повторяет 02_preprocessing.ipynb — вынесена сюда, чтобы её можно
было переиспользовать и в скриптах, и в Streamlit-приложении, без копипасты.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

LBS_TO_KG = 0.453592

ACTIVITY_ORDER = {
    "Sedentary": 0,
    "Lightly Active": 1,
    "Moderately Active": 2,
    "Very Active": 3,
}

SLEEP_ORDER = {
    "Poor": 0,
    "Fair": 1,
    "Good": 2,
    "Excellent": 3,
}

GENDER_MAP = {"F": 0, "M": 1}

NUMERIC_FEATURES = [
    "Age",
    "Current Weight (kg)",
    "Daily Calories Consumed",
    "Daily Caloric Surplus/Deficit",
    "Duration (weeks)",
    "Stress Level",
]
CATEGORICAL_FEATURES = ["Gender_encoded", "Activity_encoded", "Sleep_encoded"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Принимает df1 в исходном виде (как weight_change_dataset.csv) и
    добавляет все производные признаки — как в 02_preprocessing.ipynb.
    Не трогает Weight Change (lbs) — таргет пересчитывать здесь не нужно,
    он уже в нужных единицах на входе (lbs).
    """
    df = df.copy()

    df["Current Weight (kg)"] = df["Current Weight (lbs)"] * LBS_TO_KG
    if "Final Weight (lbs)" in df.columns:
        df["Final Weight (kg)"] = df["Final Weight (lbs)"] * LBS_TO_KG
    if "Weight Change (lbs)" in df.columns:
        df["Weight Change (kg)"] = df["Weight Change (lbs)"] * LBS_TO_KG
        df["Weight_Change_per_Week"] = df["Weight Change (lbs)"] / df["Duration (weeks)"]

    df["Deficit_Ratio"] = df["Daily Caloric Surplus/Deficit"] / df["BMR (Calories)"] \
        if "BMR (Calories)" in df.columns else None

    df["Gender_encoded"] = df["Gender"].map(GENDER_MAP)
    df["Activity_encoded"] = df["Physical Activity Level"].map(ACTIVITY_ORDER)
    df["Sleep_encoded"] = df["Sleep Quality"].map(SLEEP_ORDER)

    return df


def encode_single_input(
    age: float,
    gender: str,
    current_weight_kg: float,
    daily_calories_consumed: float,
    daily_caloric_surplus_deficit: float,
    duration_weeks: float,
    activity_level: str,
    sleep_quality: str,
    stress_level: float,
) -> pd.DataFrame:
    """
    Строит одну строку признаков (ещё БЕЗ масштабирования) для одного
    нового наблюдения, введённого пользователем в приложении.
    Возвращает DataFrame с колонками ровно в порядке MODEL_FEATURES.
    """
    row = {
        "Age": age,
        "Current Weight (kg)": current_weight_kg,
        "Daily Calories Consumed": daily_calories_consumed,
        "Daily Caloric Surplus/Deficit": daily_caloric_surplus_deficit,
        "Duration (weeks)": duration_weeks,
        "Stress Level": stress_level,
        "Gender_encoded": GENDER_MAP[gender],
        "Activity_encoded": ACTIVITY_ORDER[activity_level],
        "Sleep_encoded": SLEEP_ORDER[sleep_quality],
    }
    return pd.DataFrame([row], columns=MODEL_FEATURES)