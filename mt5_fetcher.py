import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd

def MT5DataFetcher(df: pd.DataFrame, min_years_required: int = 1):
    if not mt5.initialize():
        print("MT5 initialize() failed:", mt5.last_error())
        return df

    updated_rows = []

    for i, row in df.iterrows():
        symbol = row["Symbol"]
        if not mt5.symbol_select(symbol, True):
            print(f"❌ Can't select {symbol}")
            continue

        to_date = datetime.now()
        from_date = to_date - timedelta(days=365 * min_years_required)

        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, from_date, to_date)
        if rates is None or len(rates) == 0:
            print(f"⚠️ No MT5 data for {symbol}")
            continue

        new_data = pd.DataFrame(rates)
        new_data["Datetime"] = pd.to_datetime(new_data["time"], unit="s")
        new_data["Close"] = new_data["close"]

        latest_price = new_data["Close"].iloc[-1]
        min_coeff = (df.at[i, "Min Price"] / df.at[i, "Price"])
        max_coeff = (df.at[i, "Max Price"] / df.at[i, "Price"])
        min_price = latest_price * min_coeff
        max_price = latest_price * max_coeff

        df.at[i, "Price"] = latest_price
        df.at[i, "Min Price"] = min_price
        df.at[i, "Max Price"] = max_price

        print(f"✅ {symbol}: updated MT5 Price = {latest_price:.4f}, Min = {min_price:.4f}, Max = {max_price:.4f}")

    mt5.shutdown()
    return df