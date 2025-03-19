import pandas as pd
from scipy.stats import zscore

def analyze_365_day_gain(data: pd.DataFrame, symbol: str, std_multiplier: float):
    """Analyze average gain and std-based thresholds with separate std for gain >=0 and <0."""
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Datetime"], errors="coerce")
    data.sort_values("Date", inplace=True)

    data["Price"] = data.get("Close", data.get("Price"))
    if data["Price"].isnull().all():
        print(f"⚠️ Skipping {symbol}: 'Price' column is entirely null.")
        return data, None, None, None, None, None

    latest_row = data.dropna(subset=["Price"]).iloc[-1]
    latest_date = latest_row["Date"].date()
    latest_price = round(latest_row["Price"], 2)

    data["Price_365_Days_Later"] = data["Price"].shift(-365)
    data["Price_Gain_Percentage"] = ((data["Price_365_Days_Later"] - data["Price"]) / data["Price"]) * 100
    data.dropna(subset=["Price_Gain_Percentage"], inplace=True)
    data["Price_Gain_Percentage"] = data["Price_Gain_Percentage"].round(2)

    avg = round(data["Price_Gain_Percentage"].mean(), 2)

    # Separate stds for gain >= 0 and gain < 0
    pos_std = data.loc[data["Price_Gain_Percentage"] >= 0, "Price_Gain_Percentage"].std()
    neg_std = data.loc[data["Price_Gain_Percentage"] < 0, "Price_Gain_Percentage"].std()

    pos_std = 0.0 if pd.isna(pos_std) else pos_std
    neg_std = 0.0 if pd.isna(neg_std) else neg_std

    upper = round(avg + std_multiplier * pos_std, 2)
    lower = round(avg - std_multiplier * neg_std, 2)

    return data, avg, upper, lower, latest_date, latest_price, pos_std, neg_std

