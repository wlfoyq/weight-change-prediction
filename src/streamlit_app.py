import os
import joblib
import pandas as pd
import streamlit as st

from features import (
    encode_single_input,
    ACTIVITY_ORDER,
    SLEEP_ORDER,
    GENDER_MAP,
    LBS_TO_KG,
)
from preprocessing import transform_new_input

ACTIVITY_LABELS = {
    "Sedentary": "Малоподвижный",
    "Lightly Active": "Лёгкая активность",
    "Moderately Active": "Умеренная активность",
    "Very Active": "Высокая активность",
}

SLEEP_LABELS = {
    "Poor": "Плохое",
    "Fair": "Удовлетворительное",
    "Good": "Хорошее",
    "Excellent": "Отличное",
}

MODELS_DIR = "models"

MODEL_FILES = {
    "RandomForest (подобранная, лучшая)": "tuned_best_model.joblib",
    "RandomForest (базовые параметры)": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "Linear Regression (baseline)": "baseline_linear_regression.joblib",
    "KNN": "knn.joblib",
}

st.set_page_config(page_title="Прогноз изменения веса", page_icon="⚖️", layout="centered")


@st.cache_resource
def load_scaler():
    path = os.path.join(MODELS_DIR, "scaler.joblib")
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_resource
def load_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    return joblib.load(path) if os.path.exists(path) else None


scaler = load_scaler()

st.title("⚖️ Прогноз изменения веса")
st.caption(
    "Датасет: Comprehensive Weight Change Prediction (Kaggle), 100 наблюдений. "
    "Результат — предварительная оценка, а не точный медицинский прогноз."
)

if scaler is None:
    st.error("Не найден `models/scaler.joblib`. Сначала запустите `python train.py`.")
    st.stop()

model_label = st.selectbox("Модель для прогноза", options=list(MODEL_FILES.keys()))
model = load_model(MODEL_FILES[model_label])

if "baseline" in model_label.lower() or "linear" in model_label.lower():
    st.warning(
        "Linear Regression — самая слабая модель в этом проекте (R² < 0 на тесте). "
        "На маленькой выборке (100 наблюдений) её коэффициенты нестабильны, поэтому "
        "прогноз может быть неправдоподобно большим — это ожидаемо и наглядно показывает, "
        "почему для этой задачи выбрали ансамбли деревьев."
    )

if model is None:
    st.error(f"Не найден файл `models/{MODEL_FILES[model_label]}`. Сначала запустите `python train.py`.")
    st.stop()

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Возраст", min_value=10, max_value=100, value=30, step=1)
        gender = st.selectbox("Пол", options=list(GENDER_MAP.keys()), format_func=lambda g: "Женский" if g == "F" else "Мужской")
        current_weight_kg = st.number_input("Текущий вес (кг)", min_value=45.0, max_value=108.0, value=73.0, step=0.5)
        duration_weeks = st.number_input("Длительность наблюдения (недель)", min_value=1, max_value=104, value=12, step=1)

    with col2:
        daily_calories_consumed = st.number_input("Суточная калорийность рациона (ккал)", min_value=800.0, max_value=6000.0, value=2200.0, step=50.0)
        daily_caloric_surplus_deficit = st.number_input(
            "Суточное отклонение калорийности от нормы (ккал)",
            min_value=0.0, max_value=2000.0, value=800.0, step=50.0,
            help="В обучающих данных это всегда положительное число (82–1922 ккал) — "
                 "величина отклонения, без знака. Направление изменения веса модель "
                 "определяет по совокупности остальных признаков, а не по знаку этого поля.",
        )
        activity_level = st.selectbox("Уровень физической активности", options=list(ACTIVITY_ORDER.keys()), format_func=lambda a: ACTIVITY_LABELS[a])
        sleep_quality = st.selectbox("Качество сна", options=list(SLEEP_ORDER.keys()), format_func=lambda s: SLEEP_LABELS[s])
        stress_level = st.slider("Уровень стресса (1 — низкий, 10 — высокий)", min_value=1, max_value=10, value=5)

    submitted = st.form_submit_button("Сделать прогноз", use_container_width=True)

if submitted:
    row = encode_single_input(
        age=age,
        gender=gender,
        current_weight_kg=current_weight_kg,
        daily_calories_consumed=daily_calories_consumed,
        daily_caloric_surplus_deficit=daily_caloric_surplus_deficit,
        duration_weeks=duration_weeks,
        activity_level=activity_level,
        sleep_quality=sleep_quality,
        stress_level=stress_level,
    )
    row_scaled = transform_new_input(row, scaler)
    pred_lbs = model.predict(row_scaled)[0]
    pred_kg = pred_lbs * LBS_TO_KG

    st.divider()
    st.subheader("Результат")

    direction = "прибавка" if pred_kg > 0 else "снижение"
    st.metric(
        label=f"Прогнозируемое изменение веса за {duration_weeks} нед. ({model_label})",
        value=f"{pred_kg:+.2f} кг",
        delta=direction,
    )

with st.expander("Сравнение моделей"):
    summary_path = os.path.join("reports", "models_summary.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path).sort_values("CV_RMSE_mean").reset_index(drop=True)
        model_name_map = {
            "LinearRegression": "Линейная регрессия",
            "RandomForestRegressor": "RandomForest",
            "XGBRegressor": "XGBoost",
            "KNeighborsRegressor": "KNN",
        }
        df["Model"] = df["Model"].replace(model_name_map)
        df = df.rename(columns={
            "Model": "Модель",
            "CV_MAE_mean": "CV MAE",
            "CV_RMSE_mean": "CV RMSE",
            "CV_R2_mean": "CV R²",
            "Test_MAE": "Test MAE",
            "Test_RMSE": "Test RMSE",
            "Test_R2": "Test R²",
        })
        st.dataframe(df, use_container_width=True)
        st.caption("Метрики посчитаны на тестовой выборке в фунтах (lbs) — как обучался таргет.")
    else:
        st.write("Файл reports/models_summary.csv не найден — запустите `python train.py`.")

    importance_path = os.path.join("reports", "feature_importance.png")
    if os.path.exists(importance_path):
        st.image(importance_path, caption="Permutation importance признаков (лучшая модель)")

st.caption(
    "Наибольшее влияние на прогноз, по данным лучшей модели, оказывают уровень стресса "
    "и качество сна — не только калорийные показатели. На выборке из 100 наблюдений "
    "это стоит считать гипотезой, а не окончательным выводом."
)