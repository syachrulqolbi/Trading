import pandas as pd

def perform_eda(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        print("⚠️ DataFrame is empty. Skipping EDA summary.")
        return pd.DataFrame()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    summary = df.groupby("Symbol")["Datetime"].agg(Start_Date="min", End_Date="max")
    summary["Duration_Days"] = (summary["End_Date"] - summary["Start_Date"]).dt.days
    return summary
