"""
    python src/models/feature_importance.py
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_PATH = os.path.join("models", "tuned_best_model.joblib")
REPORTS_DIR = "reports"
IMPORTANCE_PATH = os.path.join(REPORTS_DIR, "feature_importance.xlsx")
IMPORTANCE_PLOT_PATH = os.path.join(REPORTS_DIR, "feature_importance.png")
RANDOM_STATE = 67
N_REPEATS = 30

def load_model_and_data():
    model = joblib.load(MODEL_PATH)
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X1_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X1_test.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y1_test.csv")).squeeze("columns")
    return model, X_train, X_test, y_test

def get_builtin_importance(model, feature_names) -> pd.Series:
    if not hasattr(model, "feature_importances_"):
        return None
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

def get_permutation_importance(model, X_test, y_test) -> pd.DataFrame:
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
    )
    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance_mean": result.importances_mean,
        "Importance_std": result.importances_std,
    })
    importance_df["Importance_mean"] = importance_df["Importance_mean"]
    importance_df = importance_df.sort_values("Importance_mean", ascending=False).reset_index(drop=True)
    return importance_df

def plot_importance(importance_df: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    df_sorted = importance_df.sort_values("Importance_mean")
    ax.barh(
        df_sorted["Feature"],
        df_sorted["Importance_mean"],
        xerr=df_sorted["Importance_std"],
        color="#4C72B0",
    )
    ax.set_xlabel("Рост RMSE при перемешивании признака (Permutation Importance)")
    ax.set_title("Важность признаков для прогноза изменения веса")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"\nГрафик сохранён в {path}")
    plt.close(fig)

def main():
    model, X_train, X_test, y_test = load_model_and_data()
    print(f"Модель: {type(model).__name__}")
    builtin = get_builtin_importance(model, X_train.columns)
    if builtin is not None:
        print("\n Встроенная важность признаков (feature_importances)")
        print(builtin)
    else:
        print("\n[Инфо] У этой модели нет feature_importances"
              "(это нормально для LinearRegression/KNN) — используем только Permutation Importance.")
    perm_df = get_permutation_importance(model, X_test, y_test)
    print("\n Permutation Importance (более надёжная оценка)")
    print(perm_df.to_string(index=False))
    os.makedirs(REPORTS_DIR, exist_ok=True)
    perm_df.to_excel(IMPORTANCE_PATH, index=False, sheet_name="Model Comparison")
    print(f"\nТаблица сохранена в {IMPORTANCE_PATH}")
    plot_importance(perm_df, IMPORTANCE_PLOT_PATH)
    top_3 = perm_df.head(3)["Feature"].tolist()
    print(f"\nТоп-3 самых влиятельных признака: {top_3}")
if __name__ == "__main__":
    main()
