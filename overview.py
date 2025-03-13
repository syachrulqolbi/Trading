import os
import yaml
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any
from google_sheet_api import GoogleSheetsUploader

# ----------------------------- Define Paths Dynamically -----------------------------
BASE_DIR = os.getcwd()
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
CREDENTIAL_PATH = os.path.join(BASE_DIR, "credential_google_sheets.json")

# ----------------------------- Yahoo Finance Fetcher -----------------------------
class YahooFinanceDataFetcher:
    def __init__(self, config_file: str) -> None:
        with open(config_file, "r") as file:
            self.config: Dict[str, Any] = yaml.safe_load(file)

        self.symbol_map: Dict[str, str] = self.config.get("symbols_yfinance", {})
        self.coeff_map: Dict[str, float] = self.config.get("symbol_coefficients", {})
        self.daily_period: str = self.config.get("daily_period", "10y")
        self.daily_interval: str = self.config.get("daily_interval", "1d")
        self.start_date_filter: str = self.config.get("start_date_filter", "2009-01-01")

    def fetch_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if data.empty:
                print(f"⚠️ Warning: No data available for '{ticker}'.")
            return data
        except Exception as e:
            print(f"❌ Error fetching data for '{ticker}': {e}")
            return pd.DataFrame()

    def clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if data.empty:
            return data

        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        data.rename(columns={"Date": "Datetime", "datetime": "Datetime"}, inplace=True)
        data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce", utc=True)

        data = data[data["Datetime"] >= pd.Timestamp(self.start_date_filter, tz="UTC")]
        data["Datetime"] = data["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        data["Symbol"] = symbol
        return data[["Symbol", "Datetime"] + [col for col in numeric_cols if col in data.columns]]

    def process_all_symbols(self) -> Dict[str, pd.DataFrame]:
        symbol_data = {}
        for symbol, ticker in self.symbol_map.items():
            print(f"📈 Fetching data for {symbol} ({ticker})...")
            raw_data = self.fetch_data(ticker, self.daily_period, self.daily_interval)
            if not raw_data.empty:
                cleaned_data = self.clean_data(raw_data, symbol)
                symbol_data[symbol] = cleaned_data
        if not symbol_data:
            print("⚠️ No data fetched for any symbols.")
        return symbol_data

# ----------------------------- Exploratory Data Analysis -----------------------------
def EDA(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        print("⚠️ Provided DataFrame is empty. No summary to show.")
        return pd.DataFrame()

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    summary = df.groupby("Symbol")["Datetime"].agg(Start_Date="min", End_Date="max")
    summary["Duration_Days"] = (summary["End_Date"] - summary["Start_Date"]).dt.days
    return summary

# ----------------------------- Plotting Utility -----------------------------
def plot_price_gain(data, symbol, avg, std, upper_1std, lower_1std, upper_1_97std, lower_1_97std):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))

    sns.scatterplot(data=data[data['Price_Gain_Percentage'] >= 0], x='Date', y='Price_Gain_Percentage', label='Gain >= 0%', color='green', alpha=0.6, s=10)
    sns.scatterplot(data=data[data['Price_Gain_Percentage'] < 0], x='Date', y='Price_Gain_Percentage', label='Gain < 0%', color='red', alpha=0.6, s=10)

    plt.axhline(avg, color='blue', linestyle='--', label=f'Avg Gain: {avg}%', linewidth=1.5)
    plt.axhline(upper_1std, color='purple', linestyle='--', label=f'+1 Std: {upper_1std}%', linewidth=1.2)
    plt.axhline(lower_1std, color='orange', linestyle='--', label=f'-1 Std: {lower_1std}%', linewidth=1.2)
    plt.axhline(upper_1_97std, color='darkgreen', linestyle='--', label=f'+1.97 Std: {upper_1_97std}%', linewidth=1.2)
    plt.axhline(lower_1_97std, color='darkred', linestyle='--', label=f'-1.97 Std: {lower_1_97std}%', linewidth=1.2)

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price Gain Percentage (%)', fontsize=12)
    plt.title(f'{symbol} - 365-Day Price Gain % Over Time', fontsize=16, weight='bold')
    plt.legend(loc='upper center')
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"{symbol}.jpg"), format='jpg', dpi=300)
    plt.close()

# ----------------------------- 365-Day Price Gain Analysis -----------------------------
def run_365_day_analysis(data: pd.DataFrame, symbol: str):
    data = data.copy()
    data['Date'] = pd.to_datetime(data['Datetime'], errors='coerce')
    data.sort_values('Date', inplace=True)

    data['Price'] = data.get('Close') if 'Close' in data.columns else data.get('Price')
    if data['Price'].isnull().all():
        raise ValueError("No valid 'Price' or 'Close' column found.")

    latest_row = data.dropna(subset=['Price']).iloc[-1]
    latest_date = latest_row['Date'].date()
    latest_price = round(latest_row['Price'], 2)

    data['Price_365_Days_Later'] = data['Price'].shift(-365)
    data['Price_Gain_Percentage'] = ((data['Price_365_Days_Later'] - data['Price']) / data['Price'] * 100).round(2)
    data.dropna(subset=['Price_Gain_Percentage'], inplace=True)

    avg = round(data['Price_Gain_Percentage'].mean(), 2)
    std = round(data['Price_Gain_Percentage'].std(), 2)
    upper_1std, lower_1std = round(avg + std, 2), round(avg - std, 2)
    upper_1_97std, lower_1_97std = round(avg + 1.97 * std, 2), round(avg - 1.97 * std, 2)
    std_1_97 = round(1.97 * std, 2)

    data['Std'] = std_1_97
    plot_price_gain(data, symbol, avg, std, upper_1std, lower_1std, upper_1_97std, lower_1_97std)

    return data, std_1_97, latest_date, latest_price

# ----------------------------- Negative Gain Distribution Analysis -----------------------------
def analyze_negative_gain_distribution(symbol_analysis_dict: Dict[str, pd.DataFrame]):
    print("\n📉 Analyzing negative gain distributions across symbols...")
    worst_gain_rows = []
    date_distributions = []

    for symbol, df in symbol_analysis_dict.items():
        if 'Price_Gain_Percentage' not in df.columns:
            continue
        worst_row = df.loc[df['Price_Gain_Percentage'].idxmin()]
        worst_gain_rows.append({"Symbol": symbol, "Worst Gain (%)": worst_row['Price_Gain_Percentage'], "Date of Worst Gain": worst_row['Date'].date()})
        temp_df = pd.DataFrame({"Symbol": symbol, "Negative Gain Dates": df[df['Price_Gain_Percentage'] < 0]['Date'].dt.date})
        date_distributions.append(temp_df)

    worst_gain_df = pd.DataFrame(worst_gain_rows).sort_values("Worst Gain (%)")
    print("\n🔍 Worst Gain Summary:")
    print(worst_gain_df)

    all_dist_df = pd.concat(date_distributions, ignore_index=True)
    plt.figure(figsize=(14, 8))
    sns.histplot(data=all_dist_df, x="Negative Gain Dates", hue="Symbol", multiple="stack", bins=60)
    plt.title("🗓️ Distribution of Dates with Negative Gain % by Symbol", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "negative_gain_distribution.jpg"), format='jpg', dpi=300)
    plt.close()

    return worst_gain_df

# ----------------------------- Main Execution -----------------------------
if __name__ == "__main__":
    fetcher = YahooFinanceDataFetcher(CONFIG_PATH)
    symbol_data_dict = fetcher.process_all_symbols()

    all_data_df = pd.concat(symbol_data_dict.values(), ignore_index=True)
    summary_df = EDA(all_data_df)
    print("\n📊 Symbol Data Summary:")
    print(summary_df)

    final_summary_rows = []
    symbol_analysis_dict = {}

    for symbol, df in symbol_data_dict.items():
        annotated_df, std_value, date, price = run_365_day_analysis(df, symbol)
        symbol_analysis_dict[symbol] = annotated_df
        coeff = fetcher.coeff_map.get(symbol)
        max_price = round(df["Close"].max(), 2) if "Close" in df.columns else None
        final_summary_rows.append({"Symbol": symbol, "Date": date, "Price": price, "Max Price": max_price, "Std": std_value, "Coefficient": coeff})

    final_summary_df = pd.DataFrame(final_summary_rows)
    print("\n✅ Final Summary Table (Unfiltered with Coefficients and Max Price):")
    print(final_summary_df)

    worst_gain_df = analyze_negative_gain_distribution(symbol_analysis_dict)

    try:
        print("\n📤 Uploading final summary to Google Sheets...")
        gs_uploader = GoogleSheetsUploader(CREDENTIAL_PATH, "Financial Report - Indonesia")
        gs_uploader.upload_dataframe(final_summary_df, "Overview")
        print("✅ Final summary successfully uploaded to Google Sheets!")
    except Exception as e:
        print(f"❌ Failed to upload to Google Sheets: {e}")
