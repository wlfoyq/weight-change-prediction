import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, KFold,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROCESSED_DIR = os.path.join("data", "processed")
COMPLEX_METRICS_PATH = os.path.join("reports", "complex_models_metrics.xlsx")
TUNED_METRICS_PATH = os.path.join("reports", "tuned_model_metrics.xlsx")
TUNED_MODEL_OUT_PATH = os.path.join("models", "tuned_best_model.joblib")
RANDOM_STATE = 67
CV_FOLDS = 5
SEARCH_METHOD = "grid"
def get_estimator_and_grid(model_name: str):
    if model_name == "RandomForestRegressor":
        estimator = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 3, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        }
    elif model_name == "GradientBoostingRegressor":
        estimator = GradientBoostingRegressor(random_state=RANDOM_STATE)
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.85, 1.0],
        }
    elif model_name == "XGBRegressor":
        estimator = XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.85, 1.0],
        }
    elif model_name == "KNeighborsRegressor":
        estimator = KNeighborsRegressor()
        param_grid = {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "p": [1, 2], 
        }
    return estimator, param_grid
def find_best_model_name() -> str:
    df = pd.read_excel(COMPLEX_METRICS_PATH)
    best_row = df.sort_values("CV_RMSE_mean").iloc[0]
    print(f"Лучшая модель по предыдущему шагу (CV_RMSE_mean): "
          f"{best_row['Model']} (RMSE={best_row['CV_RMSE_mean']:.4f})")
    return best_row["Model"]
def load_processed_data():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X1_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X1_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y1_train.csv")).squeeze("columns")
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y1_test.csv")).squeeze("columns")
    return X_train, X_test, y_train, y_test
def run_search(estimator, param_grid, X_train, y_train):
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    n_combinations = 1
    for v in param_grid.values():
        n_combinations *= len(v)
    print(f"\nВсего комбинаций в сетке: {n_combinations}")
    print(f"Метод подбора: {SEARCH_METHOD.upper()}"
          + (f" (проверяется {N_ITER} случайных комбинаций)" if SEARCH_METHOD == "random" else " (проверяются все комбинации)"))
    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Подбор занял {elapsed:.1f} сек.")
    print(f"Лучшие параметры: {search.best_params_}")
    print(f"Лучший CV RMSE (по метрике подбора): {-search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_

def evaluate_final_model(model, X_train, X_test, y_train, y_test) -> dict:
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_mae = -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
    cv_rmse = -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)

    return {
        "CV_MAE_mean": cv_mae.mean(), "CV_MAE_std": cv_mae.std(),
        "CV_RMSE_mean": cv_rmse.mean(), "CV_RMSE_std": cv_rmse.std(),
        "CV_R2_mean": cv_r2.mean(), "CV_R2_std": cv_r2.std(),
        "Test_MAE": test_mae, "Test_RMSE": test_rmse, "Test_R2": test_r2,
        "Train_time_sec": round(train_time, 4),
    }
def main():
    best_model_name = find_best_model_name()
    estimator, param_grid = get_estimator_and_grid(best_model_name)
    X_train, X_test, y_train, y_test = load_processed_data()
    best_estimator, best_params = run_search(estimator, param_grid, X_train, y_train)
    metrics = evaluate_final_model(best_estimator, X_train, X_test, y_train, y_test)
    print(f"\n Итоговые метрики подобранной модели ({best_model_name}, tuned) ")
    print(f"CV:   MAE={metrics['CV_MAE_mean']:.4f}±{metrics['CV_MAE_std']:.4f}  "
          f"RMSE={metrics['CV_RMSE_mean']:.4f}±{metrics['CV_RMSE_std']:.4f}  "
          f"R2={metrics['CV_R2_mean']:.4f}±{metrics['CV_R2_std']:.4f}")
    print(f"Test: MAE={metrics['Test_MAE']:.4f}  RMSE={metrics['Test_RMSE']:.4f}  "
          f"R2={metrics['Test_R2']:.4f}")
    row = {"Model": f"{best_model_name} (tuned)", **metrics}
    results_df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(TUNED_METRICS_PATH), exist_ok=True)
    results_df.to_excel(TUNED_METRICS_PATH, index=False, sheet_name="Metrics")
    print(f"\nМетрики сохранены в {TUNED_METRICS_PATH}")
    os.makedirs(os.path.dirname(TUNED_MODEL_OUT_PATH), exist_ok=True)
    joblib.dump(best_estimator, TUNED_MODEL_OUT_PATH)
    print(f"Подобранная модель сохранена в {TUNED_MODEL_OUT_PATH}")
    print(f"\nЛучшие гиперпараметры: {best_params}")
if __name__ == "__main__":
    main()
