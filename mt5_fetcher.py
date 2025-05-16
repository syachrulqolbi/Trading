import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd

def MT5DataFetcher(df: pd.DataFrame, min_years_required: int = 1):
    if not mt5.initialize():
        print("MT5 initialize() failed:", mt5.last_error())
        return df

    # Ensure Decimal column exists
    if "Decimal" not in df.columns:
        df["Decimal"] = 0  # Default value

    for i, row in df.iterrows():
        symbol = row["Symbol"]
        if not mt5.symbol_select(symbol, True):
            print(f"❌ Can't select {symbol}")
            continue

        # Retrieve Decimal Information
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ No symbol info for {symbol}")
            continue
        
        decimal_places = symbol_info.digits
        df.at[i, "Decimal"] = decimal_places

        # Fetching historical data
        to_date = datetime.now()
        from_date = to_date - timedelta(days=365 * min_years_required)

        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, from_date, to_date)
        if rates is None or len(rates) == 0:
            print(f"⚠️ No MT5 data for {symbol}")
            continue

        new_data = pd.DataFrame(rates)
        new_data["Datetime"] = pd.to_datetime(new_data["time"], unit="s")
        new_data["Close"] = new_data["close"]

        latest_price = round(new_data["Close"].iloc[-1], decimal_places)

        # Ensure the columns exist and calculate safely
        if all(col in df.columns for col in ["Min Price", "Max Price", "Price"]):
            min_coeff = (df.at[i, "Min Price"] / df.at[i, "Price"]) if df.at[i, "Price"] != 0 else 0
            max_coeff = (df.at[i, "Max Price"] / df.at[i, "Price"]) if df.at[i, "Price"] != 0 else 0
            
            min_price = round(latest_price * min_coeff, decimal_places)
            max_price = round(latest_price * max_coeff, decimal_places)

            df.at[i, "Price"] = latest_price
            df.at[i, "Min Price"] = min_price
            df.at[i, "Max Price"] = max_price

            print(f"✅ {symbol}: updated MT5 Price = {latest_price:.{decimal_places}f}, Min = {min_price:.{decimal_places}f}, Max = {max_price:.{decimal_places}f}")
        else:
            print(f"⚠️ Missing Price columns for {symbol}. Skipping price update.")

    mt5.shutdown()
    return df