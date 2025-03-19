import pandas as pd

def analyze_max_negative_gain(data: pd.DataFrame, symbol: str):
    """Analyze maximum negative gain from entry point to any future point (unlimited future days)."""
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Datetime"], errors="coerce")
    data.sort_values("Date", inplace=True)

    data["Price"] = data.get("Close", data.get("Price"))
    if data["Price"].isnull().all():
        print(f"⚠️ Skipping {symbol}: 'Price' column is entirely null.")
        return data, None, None, None, None

    latest_row = data.dropna(subset=["Price"]).iloc[-1]
    latest_date = latest_row["Date"].date()
    latest_price = round(latest_row["Price"], 2)

    # Calculate worst possible drop from each row to any point in the future
    worst_losses = []
    prices = data["Price"].values

    for i in range(len(prices)):
        entry_price = prices[i]
        future_prices = prices[i + 1:]

        if len(future_prices) == 0:
            worst_losses.append(None)
            continue

        future_returns = ((future_prices - entry_price) / entry_price) * 100
        worst_losses.append(round(future_returns.min(), 2))

    data["Worst_Future_Gain_%"] = worst_losses

    worst_losses_clean = [x for x in worst_losses if x is not None]
    worst_drawdown = round(pd.Series(worst_losses_clean).quantile(0.33), 2)
    max_drawdown = round(min(worst_losses_clean), 2)

    return data, worst_drawdown, max_drawdown, latest_date, latest_price
