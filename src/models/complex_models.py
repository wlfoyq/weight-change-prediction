import os
import time
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor

PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_PATH = os.path.join("reports", "complex_models_metrics.xlsx")
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


def build_models() -> dict:
    models = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "KNeighborsRegressor": KNeighborsRegressor(
            n_neighbors=5,
        ),
    }
    models["XGBRegressor"] = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            verbosity=0,
    )
    return models

def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
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
def compare_models(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    results = []
    for name, model in models.items():
        print(f"\n {name}")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        metrics["Model"] = name
        results.append(metrics)
        print(
            f"  CV(5-fold):  MAE={metrics['CV_MAE_mean']:.4f}±{metrics['CV_MAE_std']:.4f}  "
            f"RMSE={metrics['CV_RMSE_mean']:.4f}±{metrics['CV_RMSE_std']:.4f}  "
            f"R2={metrics['CV_R2_mean']:.4f}±{metrics['CV_R2_std']:.4f}"
        )
        print(
            f"  Test (hold-out): MAE={metrics['Test_MAE']:.4f}  "
            f"RMSE={metrics['Test_RMSE']:.4f}  R2={metrics['Test_R2']:.4f}"
        )
    cols = ["Model", "CV_MAE_mean", "CV_MAE_std", "CV_RMSE_mean", "CV_RMSE_std",
            "CV_R2_mean", "CV_R2_std", "Test_MAE", "Test_RMSE", "Test_R2", "Train_time_sec"]
    results_df = pd.DataFrame(results)[cols]
    results_df = results_df.sort_values("CV_RMSE_mean").reset_index(drop=True)
    return results_df

def print_feature_importance(model, feature_names, model_name):
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
    print(f"\nВажность признаков ({model_name}):")
    print(importances)

def main():
    X_train, X_test, y_train, y_test = load_processed_data()
    models = build_models()
    results_df = compare_models(models, X_train, X_test, y_train, y_test)
    print("\n Итоговая таблица сравнения моделей ")
    print(results_df.to_string(index=False))
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results_df.to_excel(RESULTS_PATH, index=False, sheet_name="Metrics")
    print(f"\nТаблица сохранена в {RESULTS_PATH}")
    rf_model = models["RandomForestRegressor"]
    print_feature_importance(rf_model, X_train.columns, "RandomForestRegressor")
if __name__ == "__main__":
    main()
