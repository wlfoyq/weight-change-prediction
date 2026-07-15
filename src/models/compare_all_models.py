import os
import pandas as pd

REPORTS_DIR = "reports"
BASELINE_METRICS_PATH = os.path.join(REPORTS_DIR, "baseline_metrics.xlsx")
COMPLEX_METRICS_PATH = os.path.join(REPORTS_DIR, "complex_models_metrics.xlsx")
TUNED_METRICS_PATH = os.path.join(REPORTS_DIR, "tuned_model_metrics.xlsx")
FINAL_TABLE_XLSX_PATH = os.path.join(REPORTS_DIR, "model_comparison.xlsx")

def load_metrics(path: str, source_name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Не найден файл {path}."
        )
    return pd.read_excel(path)

def main():
    baseline_df = load_metrics(BASELINE_METRICS_PATH, "baseline_model.py")
    complex_df = load_metrics(COMPLEX_METRICS_PATH, "complex_models.py")
    frames = [baseline_df, complex_df]
    tuned_df = pd.read_excel(TUNED_METRICS_PATH)
    frames.append(tuned_df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("CV_RMSE_mean").reset_index(drop=True)
    combined.insert(0, "Rank", range(1, len(combined) + 1))
    print(" Итоговая таблица сравнения всех моделей \n")
    print(combined.to_string(index=False))
    os.makedirs(REPORTS_DIR, exist_ok=True)
    combined.round(4).to_excel(FINAL_TABLE_XLSX_PATH, index=False, sheet_name="Model Comparison")
    print(f"Excel-файл сохранён в {FINAL_TABLE_XLSX_PATH}")
    best_model = combined.iloc[0]["Model"]
    print(f"\nЛучшая модель по CV_RMSE_mean: {best_model}")

if __name__ == "__main__":
    main()