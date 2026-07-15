import os
import joblib
import numpy as np
import pandas as pd
import time

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_OUT_PATH = os.path.join("models", "baseline_linear_regression.joblib")
RESULTS_PATH = os.path.join("reports", "baseline_metrics.xlsx")
RANDOM_STATE = 67
CV_FOLDS = 5

def load_processed_data():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X1_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X1_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y1_train.csv")).squeeze("columns")
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y1_test.csv")).squeeze("columns")

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Признаки: {list(X_train.columns)}")
    return X_train, X_test, y_train, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = LinearRegression()

    # --- 1. Кросс-валидация на train (cv=5)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_mae = -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
    cv_rmse = -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")

    # --- 2. Финальное обучение на всём train + проверка на test
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)

    print("\n CV (5-fold) метрики baseline-модели (Linear Regression)")
    print(f"MAE  : {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
    print(f"RMSE : {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    print(f"R^2  : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    print("\n Test (hold-out) метрики baseline-модели")
    print(f"MAE  : {test_mae:.4f}")
    print(f"RMSE : {test_rmse:.4f}")
    print(f"R^2  : {test_r2:.4f}")

    metrics = {
        "CV_MAE_mean": cv_mae.mean(), "CV_MAE_std": cv_mae.std(),
        "CV_RMSE_mean": cv_rmse.mean(), "CV_RMSE_std": cv_rmse.std(),
        "CV_R2_mean": cv_r2.mean(), "CV_R2_std": cv_r2.std(),
        "Test_MAE": test_mae, "Test_RMSE": test_rmse, "Test_R2": test_r2,
        "Train_time_sec": round(train_time, 4),
    }
    return model, metrics

def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"\nМодель сохранена в {path}")

def save_metrics(metrics: dict, path: str):
    row = {"Model": "LinearRegression", **metrics}
    results_df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results_df.to_excel(path, index=False, sheet_name="Metrics")
    print(f"Метрики сохранены в {path}")
def main():
    X_train, X_test, y_train, y_test = load_processed_data()
    model, metrics = train_and_evaluate(X_train, X_test, y_train, y_test)
    save_model(model, MODEL_OUT_PATH)
    save_metrics(metrics, RESULTS_PATH)
if __name__ == "__main__":
    main()
