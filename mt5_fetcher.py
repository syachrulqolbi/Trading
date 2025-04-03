import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd

def MT5DataFetcher(df, symbol_list):
    if not mt5.initialize():
        print("MT5 initialize() failed:", mt5.last_error())
        return df

    for symbol in symbol_list:
        if not mt5.symbol_select(symbol, True):
            print(f"❌ Can't select {symbol}")
            continue

        # Get the last date for this symbol in the DataFrame
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        last_date = df[df["Symbol"] == symbol]["Datetime"].max()
        print(f"Last date for {symbol}: {last_date}")

        # If there's no previous data, start from 30 days ago
        from_date = last_date - timedelta(days=365) if pd.notnull(last_date) else datetime.now() - timedelta(days=30)
        to_date = last_date

        # Fetch daily data from MT5
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, from_date, to_date)
        if rates is None or len(rates) == 0:
            print(f"⚠️ No daily data for {symbol}")
            continue

        # Convert to DataFrame
        new_data = pd.DataFrame(rates)
        new_data["Datetime"] = pd.to_datetime(new_data["time"], unit='s')
        new_data["Symbol"] = symbol
        new_data.rename(columns={"close": "Close"}, inplace=True)

        # Only keep the required columns to match your original df
        new_data = new_data[["Datetime", "Symbol", "Close"]]
        update_date = new_data["Datetime"].values[-1]
        update_value = new_data["Close"].values[-1]
        df.loc[(df["Symbol"] == symbol) & (df["Datetime"] == update_date), "Close"] = update_value
        print(f"Updated latest close price for {symbol} on {update_date}.")

        # Append new data and remove duplicates
        df = pd.concat([df, new_data], ignore_index=True)

        # Remove all data after the last MT5 date for this symbol
        df = df.drop(df[(df["Symbol"] == symbol) & (df["Datetime"] > update_date)].index)

        # Remove duplicates and reset the index
        df = df.drop_duplicates(subset=["Datetime", "Symbol"], keep="last").reset_index(drop=True)

    mt5.shutdown()
    return df