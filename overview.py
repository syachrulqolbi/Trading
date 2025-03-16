import os
import yaml
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any
from google_sheet_api import GoogleSheetsUploader
from scipy.stats import zscore

# === Configuration and Directory Setup ===
BASE_DIR = os.getcwd()
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
CREDENTIAL_PATH = os.path.join(BASE_DIR, "credential_google_sheets.json")


def cleanup_existing_plots(plot_name: str):
    """Delete existing plot image if exists."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, plot_name)
    if os.path.exists(plot_path):
        os.remove(plot_path)


# === Yahoo Finance Data Fetcher Class ===
class YahooFinanceDataFetcher:
    def __init__(self, config_file: str) -> None:
        with open(config_file, "r") as file:
            self.config: Dict[str, Any] = yaml.safe_load(file)
        self.symbol_map = self.config.get("symbols_yfinance", {})
        self.coeff_map = self.config.get("symbol_coefficients", {})
        self.daily_period = self.config.get("daily_period", "10y")
        self.daily_interval = self.config.get("daily_interval", "1d")
        self.std_multiplier = float(self.config.get("std_multiplier", 1.97))

    def fetch_data(self, ticker: str) -> pd.DataFrame:
        """Download historical price data using yfinance."""
        try:
            data = yf.download(ticker, period=self.daily_period, interval=self.daily_interval, progress=False)
            if data.empty:
                print(f"⚠️ No data for '{ticker}'.")
            return data
        except Exception as e:
            print(f"❌ Error fetching '{ticker}': {e}")
            return pd.DataFrame()

    def clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and standardize data format."""
        if data.empty:
            return data
        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        data.rename(columns={"Date": "Datetime", "datetime": "Datetime"}, inplace=True)
        data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        data["Symbol"] = symbol
        return data[["Symbol", "Datetime"] + [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in data.columns]]

    def process_all_symbols(self) -> Dict[str, pd.DataFrame]:
        """Fetch and clean data for all configured symbols."""
        symbol_data = {}
        for symbol, ticker in self.symbol_map.items():
            print(f"📈 Fetching {symbol} ({ticker})...")
            raw_data = self.fetch_data(ticker)
            if not raw_data.empty:
                symbol_data[symbol] = self.clean_data(raw_data, symbol)
        if not symbol_data:
            print("⚠️ No data fetched for any symbols.")
        return symbol_data


# === Exploratory Data Analysis (EDA) ===
def perform_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Summary of available date range and duration for each symbol."""
    if df.empty:
        print("⚠️ DataFrame is empty. Skipping EDA summary.")
        return pd.DataFrame()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    summary = df.groupby("Symbol")["Datetime"].agg(Start_Date="min", End_Date="max")
    summary["Duration_Days"] = (summary["End_Date"] - summary["Start_Date"]).dt.days
    return summary

def plot_price_gain(data, symbol, avg, std, upper_1std, lower_1std, upper_1_97std, lower_1_97std):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))

    sns.scatterplot(data=data[data['Price_Gain_Percentage'] >= 0], x='Date', y='Price_Gain_Percentage', label='Gain ≥ 0%', color='green', alpha=0.6, s=10)
    sns.scatterplot(data=data[data['Price_Gain_Percentage'] < 0], x='Date', y='Price_Gain_Percentage', label='Gain < 0%', color='red', alpha=0.6, s=10)

    plt.axhline(avg, color='blue', linestyle='--', label=f'Avg Gain: {avg}%')
    plt.axhline(upper_1std, color='purple', linestyle='--', label=f'+1 Std: {upper_1std}%')
    plt.axhline(lower_1std, color='orange', linestyle='--', label=f'-1 Std: {lower_1std}%')
    plt.axhline(upper_1_97std, color='darkgreen', linestyle='--', label=f'+1.97 Std: {upper_1_97std}%')
    plt.axhline(lower_1_97std, color='darkred', linestyle='--', label=f'-1.97 Std: {lower_1_97std}%')

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('365-Day Gain Percentage (%)', fontsize=12)
    plt.title(f'{symbol} - 365-Day Price Gain % Over Time', fontsize=16)
    plt.legend(loc='upper center')
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"{symbol}_gain_plot.jpg"), format='jpg', dpi=300)
    plt.close()

# === 365-Day Gain Analysis ===
def analyze_365_day_gain(data: pd.DataFrame, symbol: str, std_multiplier: float):
    """Calculate average gain and adjusted standard deviation (excluding outliers above +2.576 std)."""
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

    # ✅ Remove outliers > +2.576 std for std calculation only
    z_scores = zscore(data["Price_Gain_Percentage"])
    filtered_data = data[(z_scores <= std_multiplier)]  # keeping everything within Z <= 2.576
    filtered_std = round(filtered_data["Price_Gain_Percentage"].std(), 2)

    upper = round(avg + std_multiplier * filtered_std, 2)
    lower = round(avg - std_multiplier * filtered_std, 2)

    return data, avg, upper, lower, latest_date, latest_price


# === Weekly Backtest Simulation ===
def backtest_weekly_investment(df: pd.DataFrame, initial_balance: float, invest_per_week: float, tp_percent: float,
                                leverage: float, coeff: float, std: float, start_date: str = None, end_date: str = None):
    """Simulate a weekly investment portfolio with take profit threshold."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Datetime"]).dt.date
    df = df.sort_values("Date")

    if start_date:
        df = df[df["Date"] >= pd.to_datetime(start_date).date()]
    if end_date:
        df = df[df["Date"] <= pd.to_datetime(end_date).date()]

    df["Week"] = pd.to_datetime(df["Date"]).dt.to_period("W").apply(lambda r: r.start_time.date())
    weekly_df = df.groupby("Week").first().reset_index()
    weekly_df = weekly_df[weekly_df["Close"].notnull()]

    cash_invest = cash_saving = cash_saving_interest = initial_balance
    list_trade_price, list_lot_size = [], []
    portfolio_history = []

    for _, row in weekly_df.iterrows():
        price = row["Close"]
        if pd.isna(price) or price <= 0:
            continue

        value_divider = std * price * coeff
        lot_size = max(round(cash_invest / value_divider, 2), 0.01)
        profit_tp = 0.0

        if not list_trade_price:
            list_trade_price.append(price)
            list_lot_size.append(lot_size)
            cash_invest += invest_per_week
        else:
            previous_price = list_trade_price[-1]
            if previous_price <= price * (1 + tp_percent / 100.0):
                avg_trade_price = sum(p * l for p, l in zip(list_trade_price, list_lot_size)) / sum(list_lot_size)
                profit_tp = avg_trade_price * sum(list_lot_size) * (tp_percent / 100.0) * coeff * 100 * leverage / 1000
                cash_invest += invest_per_week + profit_tp
                list_trade_price, list_lot_size = [], []
            else:
                list_trade_price.append(price)
                list_lot_size.append(lot_size)
                cash_invest += invest_per_week
                avg_trade_price = sum(p * l for p, l in zip(list_trade_price, list_lot_size)) / sum(list_lot_size)
                if price < avg_trade_price * (1 - std / 100.0):
                    print(f"🚨 {df['Symbol'].iloc[0] if 'Symbol' in df.columns else ''}: Price dropped below avg trade price - std at {row['Week']}. Portfolio wiped out.")
                    cash_invest = 0
                    break

        cash_saving += invest_per_week
        weekly_interest_rate = (1 + 0.05) ** (1 / 52) - 1
        cash_saving_interest = (cash_saving_interest + invest_per_week) * (1 + weekly_interest_rate)

        portfolio_history.append({
            "Week": row["Week"], "Close_Price": price, "Profit_TP": round(profit_tp, 2),
            "Cash_Invest": cash_invest, "Cash_Saving": cash_saving,
            "Cash_Saving_Interest": cash_saving_interest
        })

    portfolio_df = pd.DataFrame(portfolio_history)

    def calculate_adjusted_return(df: pd.DataFrame, value_col: str) -> float:
        if df.empty or value_col not in df.columns:
            return 0.0
        years = (df["Week"].iloc[-1] - df["Week"].iloc[0]).days / 365.25
        if years <= 0: return 0.0
        total_contribution = initial_balance + invest_per_week * len(df)
        final_value = df[value_col].iloc[-1]
        return round(((final_value / total_contribution) ** (1 / years) - 1) * 100, 2)

    ar_invest = calculate_adjusted_return(portfolio_df, "Cash_Invest")
    ar_saving = calculate_adjusted_return(portfolio_df, "Cash_Saving")
    ar_saving_interest = calculate_adjusted_return(portfolio_df, "Cash_Saving_Interest")

    # === Save Plot ===
    symbol_title = df["Symbol"].iloc[0] if "Symbol" in df.columns else "Symbol"
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_df['Week'], portfolio_df['Cash_Invest'], label=f'Investment (TP {tp_percent}% | AR {ar_invest}%)', linestyle='-.')
    plt.plot(portfolio_df['Week'], portfolio_df['Cash_Saving'], label=f'Saving (AR {ar_saving}%)', linestyle='--')
    plt.plot(portfolio_df['Week'], portfolio_df['Cash_Saving_Interest'], label=f'Saving +5% Interest (AR {ar_saving_interest}%)', linestyle=':')
    plt.title(f'{symbol_title} - Weekly Investment vs Saving', fontsize=14)
    plt.xlabel('Week'); plt.ylabel('Total Value ($)')
    plt.legend(); plt.xticks(rotation=45); plt.grid(True); plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, f"{symbol_title}_investment_plot.jpg")
    plt.savefig(plot_path)
    plt.close()

    return portfolio_df, ar_invest, ar_saving, ar_saving_interest


# === Main Execution ===
if __name__ == "__main__":
    fetcher = YahooFinanceDataFetcher(CONFIG_PATH)
    symbol_data = fetcher.process_all_symbols()
    full_df = pd.concat(symbol_data.values(), ignore_index=True)

    print("\n📊 EDA Summary:")
    print(perform_eda(full_df))

    final_summary, analyzed_data = [], {}
    for symbol, df in symbol_data.items():
        annotated_df, avg, upper, lower, latest_dt, latest_price = analyze_365_day_gain(df, symbol, fetcher.std_multiplier)
        if avg is None:
            continue
        
        std = round((upper - avg) / fetcher.std_multiplier, 2)
        upper_1std = round(avg + std, 2)
        lower_1std = round(avg - std, 2)
        upper_1_97std = round(avg + 1.97 * std, 2)
        lower_1_97std = round(avg - 1.97 * std, 2)

        plot_price_gain(annotated_df, symbol, avg, std, upper_1std, lower_1std, upper_1_97std, lower_1_97std)

        analyzed_data[symbol] = annotated_df
        # Before filtering
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", utc=True)
        ten_years_ago = pd.Timestamp.now(tz='UTC') - pd.DateOffset(years=10)

        # Safe comparison
        max_price = round(df[df['Datetime'] >= ten_years_ago]["Close"].max(), 2) if "Close" in df.columns else None

        portfolio_df, ar_invest, _, _ = backtest_weekly_investment(
            df, initial_balance=0, invest_per_week=200, tp_percent=1.0,
            leverage=1000,
            coeff=fetcher.coeff_map.get(symbol),
            std=abs(lower),
            start_date="1900-01-01", end_date="2024-12-31"
        )

        final_summary.append({
            "Symbol": symbol, "Date": latest_dt, "Price": latest_price, "Max Price": max_price,
            "Std": abs(lower), "Coefficient": fetcher.coeff_map.get(symbol), "Annual Return (Simulated)": ar_invest
        })

    final_df = pd.DataFrame(final_summary)

    print("\n✅ Final Summary:")
    print(final_df)
    
    # Upload to Google Sheets
    try:
        print("\n📤 Uploading to Google Sheets...")
        uploader = GoogleSheetsUploader(CREDENTIAL_PATH, "Financial Report - Indonesia")
        uploader.upload_dataframe(final_df, "Overview")
        print("✅ Upload successful!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")