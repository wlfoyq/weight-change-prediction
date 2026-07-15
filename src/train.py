import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from preprocessing import fit_scaler_on_raw_data

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
RANDOM_STATE = 67
CV_FOLDS = 5

TUNING_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}


def load_processed_data():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X1_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X1_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y1_train.csv")).squeeze("columns")
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y1_test.csv")).squeeze("columns")
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, X_test, y_train, y_test):
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_mae = -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
    cv_rmse = -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)

    return {
        "CV_MAE_mean": cv_mae.mean(), "CV_RMSE_mean": cv_rmse.mean(), "CV_R2_mean": cv_r2.mean(),
        "Test_MAE": test_mae, "Test_RMSE": test_rmse, "Test_R2": test_r2,
    }


def train_base_models(X_train, X_test, y_train, y_test):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBRegressor": XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE, verbosity=0),
        "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5),
    }

    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        metrics["Model"] = name
        results.append(metrics)
        print(f"{name}: CV_RMSE={metrics['CV_RMSE_mean']:.4f}  Test_R2={metrics['Test_R2']:.4f}")

    return models, pd.DataFrame(results)


def tune_best_model(X_train, X_test, y_train, y_test):
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=TUNING_PARAM_GRID,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    print(f"Лучшие параметры (tuned RandomForest): {search.best_params_}")

    model = search.best_estimator_
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    print(f"Tuned RandomForest: CV_RMSE={metrics['CV_RMSE_mean']:.4f}  Test_R2={metrics['Test_R2']:.4f}")
    return model


def main():
    X_train, X_test, y_train, y_test = load_processed_data()

    models, comparison_df = train_base_models(X_train, X_test, y_train, y_test)
    tuned_model = tune_best_model(X_train, X_test, y_train, y_test)
    scaler = fit_scaler_on_raw_data()

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(models["LinearRegression"], os.path.join(MODELS_DIR, "baseline_linear_regression.joblib"))
    joblib.dump(models["RandomForestRegressor"], os.path.join(MODELS_DIR, "random_forest.joblib"))
    joblib.dump(models["XGBRegressor"], os.path.join(MODELS_DIR, "xgboost.joblib"))
    joblib.dump(models["KNeighborsRegressor"], os.path.join(MODELS_DIR, "knn.joblib"))
    joblib.dump(tuned_model, os.path.join(MODELS_DIR, "tuned_best_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    os.makedirs("reports", exist_ok=True)
    comparison_df.to_csv(os.path.join("reports", "models_summary.csv"), index=False)

    print(f"Все модели и scaler сохранены в {MODELS_DIR}/")


if __name__ == "__main__":
    main()