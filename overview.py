import os
import pandas as pd
from datetime import datetime
from data_fetcher import YahooFinanceDataFetcher
from eda_utils import perform_eda
from analyzer import analyze_365_day_gain
from backtester import backtest_weekly_investment
from plot_utils import plot_price_gain
from google_sheet_api import GoogleSheetsUploader

# === Setup ===
BASE_DIR = os.getcwd()
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
CREDENTIAL_PATH = os.path.join(BASE_DIR, "credential_google_sheets.json")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

def main():
    fetcher = YahooFinanceDataFetcher(CONFIG_PATH)
    symbol_data = fetcher.process_all_symbols()
    full_df = pd.concat(symbol_data.values(), ignore_index=True)

    # --- EDA Summary ---
    print("\n📊 EDA Summary:")
    print(perform_eda(full_df))

    final_summary, analyzed_data = [], {}

    for symbol, df in symbol_data.items():
        annotated_df, avg, upper, lower, latest_dt, latest_price = analyze_365_day_gain(
            df, symbol, fetcher.std_multiplier
        )

        if avg is None:
            continue

        std = round((upper - avg) / fetcher.std_multiplier, 2)
        upper_1std = round(avg + std, 2)
        lower_1std = round(avg - std, 2)
        upper_1_97std = round(avg + 1.97 * std, 2)
        lower_1_97std = round(avg - 1.97 * std, 2)

        # Plot price gain distribution
        plot_price_gain(annotated_df, symbol, avg, std, upper_1std, lower_1std, upper_1_97std, lower_1_97std)

        analyzed_data[symbol] = annotated_df

        # Compute max price (past 10 years)
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", utc=True)
        ten_years_ago = pd.Timestamp.now(tz='UTC') - pd.DateOffset(years=10)
        max_price = round(df[df["Datetime"] >= ten_years_ago]["Close"].max(), 2) if "Close" in df.columns else None

        # Simulate backtest
        portfolio_df, ar_invest, _, _ = backtest_weekly_investment(
            df,
            initial_balance=0,
            invest_per_week=200,
            tp_percent=1.0,
            leverage=1000,
            coeff=fetcher.coeff_map.get(symbol),
            std=abs(lower),
            start_date="1900-01-01",
            end_date="2024-12-31"
        )

        final_summary.append({
            "Symbol": symbol,
            "Date": latest_dt,
            "Price": latest_price,
            "Max Price": max_price,
            "Std": abs(lower),
            "Coefficient": fetcher.coeff_map.get(symbol),
            "Annual Return (Simulated)": ar_invest
        })

    # --- Final Summary ---
    final_df = pd.DataFrame(final_summary)
    print("\n✅ Final Summary:")
    print(final_df)

    # --- Upload to Google Sheets ---
    try:
        print("\n📤 Uploading to Google Sheets...")
        uploader = GoogleSheetsUploader(CREDENTIAL_PATH, "Financial Report - Indonesia")
        uploader.upload_dataframe(final_df, "Overview")
        print("✅ Upload successful!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    main()