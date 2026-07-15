import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

from features import engineer_features, NUMERIC_FEATURES, MODEL_FEATURES

RAW_DATA_PATH = os.path.join("data", "raw", "weight_change_dataset.csv")


def fit_scaler_on_raw_data(raw_path: str = RAW_DATA_PATH) -> StandardScaler:
    df1 = pd.read_csv(raw_path)
    df1 = engineer_features(df1)
    scaler = StandardScaler()
    scaler.fit(df1[NUMERIC_FEATURES])
    return scaler


def transform_new_input(row_df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    row_df = row_df.copy()
    row_df[NUMERIC_FEATURES] = scaler.transform(row_df[NUMERIC_FEATURES])
    return row_df[MODEL_FEATURES]