import pandas as pd
from datetime import timedelta
from plot_utils import plot_analysis
from datetime import timedelta

def analyze_drawdown_and_gain(data: pd.DataFrame, symbol: str, min_years_rquired: int):
    """
    Analyze maximum drawdown and maximum gain within the next 10 years from each entry point.
    Adds 'Max_Drawdown' and 'Max_Gain' columns to DataFrame.
    """
    if data.empty or "Datetime" not in data.columns:
        print(f"⚠️ Skipping {symbol}: missing or empty data.")
        return data

    data["Date"] = pd.to_datetime(data["Datetime"], errors="coerce")
    data.sort_values("Date", inplace=True)
    data.reset_index(drop=True, inplace=True)

    data["Price"] = data.get("Close") if "Close" in data.columns else data.get("Price")
    if data["Price"].isnull().all():
        print(f"⚠️ Skipping {symbol}: 'Price' column is entirely null.")
        return data

    last_date = data["Date"].max()
    cutoff_date = last_date - timedelta(days=365 * min_years_rquired)

    max_drawdowns, max_gains = [], []

    for i in range(len(data)):
        entry_date = data.loc[i, "Date"]
        if entry_date > cutoff_date:
            max_drawdowns.append(None)
            max_gains.append(None)
            continue

        entry_price = data.loc[i, "Price"]
        if pd.isna(entry_price) or entry_price <= 0:
            max_drawdowns.append(None)
            max_gains.append(None)
            continue

        end_date = entry_date + timedelta(days=365 * min_years_rquired)
        future_prices = data[(data["Date"] > entry_date) & (data["Date"] <= end_date)]["Price"]

        # Remove invalid prices (zero or negative)
        future_prices = future_prices[future_prices > 0]

        if future_prices.empty:
            max_drawdowns.append(None)
            max_gains.append(None)
            continue

        returns = ((future_prices - entry_price) / entry_price) * 100
        max_drawdowns.append(round(returns.min(), 2))
        max_gains.append(round(returns.max(), 2))

    data["Max_Drawdown"] = max_drawdowns
    data["Max_Gain"] = max_gains

    return data

def backtest_weekly_investment(
    df: pd.DataFrame,
    initial_balance: float,
    invest_per_week: float,
    tp_percent: float,
    leverage: float,
    coeff: float,
    std: float,
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    Simulates a daily backtest with weekly investment deposits.
    Returns a DataFrame with portfolio evolution and key metrics.
    """
    # --- Preprocessing ---
    df["Date"] = pd.to_datetime(df["Datetime"]).dt.date
    df = df.sort_values("Date")

    if start_date:
        df = df[df["Date"] >= pd.to_datetime(start_date).date()]
    if end_date:
        df = df[df["Date"] <= pd.to_datetime(end_date).date()]

    df["Week"] = pd.to_datetime(df["Date"]).dt.to_period("W").apply(lambda r: r.start_time.date())
    
    # --- Initialization ---
    cash_invest = cash_saving = cash_saving_interest = initial_balance
    trade_prices, lot_sizes = [], []
    history = []

    current_week = None
    weekly_rate = (1 + 0.05) ** (1 / 52) - 1
    stopped_early = False 

    for _, row in df.iterrows():
        price = row["Close"]
        date = row["Date"]
        week = row["Week"]

        if pd.isna(price) or price <= 0:
            continue

        # New week: Add investment and update savings
        if week != current_week:
            current_week = week
            cash_invest += invest_per_week
            cash_saving += invest_per_week
            cash_saving_interest = (cash_saving_interest + invest_per_week) * (1 + weekly_rate)

        # Calculate lot size based on current available cash and volatility-adjusted risk
        value_divider = std * price * coeff
        lot_size = max(round(cash_invest / value_divider, 2), 0.01)
        profit_tp = 0.0

        if not trade_prices:
            trade_prices.append(price)
            lot_sizes.append(lot_size)
        else:
            last_trade_price = trade_prices[-1]
            target_price = last_trade_price * (1 + tp_percent / 100)

            if price >= target_price:
                avg_trade_price = sum(p * l for p, l in zip(trade_prices, lot_sizes)) / sum(lot_sizes)
                profit_tp = avg_trade_price * sum(lot_sizes) * (tp_percent / 100) * coeff * 100 * leverage / 1000
                cash_invest += profit_tp
                trade_prices.clear()
                lot_sizes.clear()
            else:
                trade_prices.append(price)
                lot_sizes.append(lot_size)

                # Stop-loss: price drops below avg - std
                avg_trade_price = sum(p * l for p, l in zip(trade_prices, lot_sizes)) / sum(lot_sizes)
                if price < avg_trade_price * (1 - std / 100.0):
                    stopped_early = True 
                    break  # Stop trading due to loss
        
        avg_lot = round(sum(lot_sizes) / len(lot_sizes), 4) if lot_sizes else 0.0

        history.append({
            "Date": date,
            "Week": week,
            "Close_Price": price,
            "Profit_TP": round(profit_tp, 2),
            "Cash_Invest": round(cash_invest, 2),
            "Cash_Saving": round(cash_saving, 2),
            "Cash_Saving_Interest": round(cash_saving_interest, 2),
            "Average_Lot_Size": avg_lot
        })

    result_df = pd.DataFrame(history)

    # --- Adjusted Return Calculations ---
    def calc_ar(value_col: str) -> float:
        if result_df.empty or value_col not in result_df.columns:
            return 0.0
        duration_years = (result_df["Date"].iloc[-1] - result_df["Date"].iloc[0]).days / 365.25
        if duration_years <= 0:
            return 0.0
        total_contribution = initial_balance + invest_per_week * len(result_df["Week"].drop_duplicates())
        final_value = result_df[value_col].iloc[-1]
        return round(((final_value / total_contribution) ** (1 / duration_years) - 1) * 100, 2)

    ar_invest = calc_ar("Cash_Invest")
    ar_saving = calc_ar("Cash_Saving")
    ar_saving_interest = calc_ar("Cash_Saving_Interest")

    result_df["AR_Invest"] = ar_invest
    result_df["AR_Saving"] = ar_saving
    result_df["AR_Saving_Interest"] = ar_saving_interest

    return result_df, ar_invest, ar_saving, ar_saving_interest, stopped_early

def run_analysis(
    df: pd.DataFrame,
    symbol: str,
    std_multiplier: float,
    plots_dir: str = None,
    tp_percent: float = 1.0,
    leverage: float = 1000,
    coeff: float = 0.01,
    initial_balance: float = 1000,
    invest_per_week: float = 10,
    min_years_required: int = 10
):
    df = df.loc[(df["Symbol"] == symbol)].copy()

    df["Date"] = pd.to_datetime(df["Datetime"]).dt.date
    latest_date = df["Date"].max()
    start_date = (latest_date - timedelta(days=365 * min_years_required)).strftime("%Y-%m-%d 00:00:00")
    end_date = latest_date.strftime("%Y-%m-%d 00:00:00")

    if df.empty:
        print(f"⚠️ Skipping {symbol}: no data found.")
        return None, None, None, None

    # Check date coverage
    df["Date"] = pd.to_datetime(df["Datetime"]).dt.date
    date_range_years = (df["Date"].max() - df["Date"].min()).days / 365.25

    if date_range_years < min_years_required:
        print(f"⏭️ {symbol}: only {round(date_range_years, 2)} years of data (< {min_years_required} years).")
        min_years_required = (df["Date"].max() - df["Date"].min()).days / 365.25

    data = analyze_drawdown_and_gain(df, symbol, min_years_required)
    data = data.dropna(subset=["Max_Drawdown", "Max_Gain"])
    if data.empty:
        print(f"⚠️ No valid drawdown/gain data for {symbol}.")
        return data, None, None

    # Thresholds
    dd_mean, dd_std = data["Max_Drawdown"].mean(), data["Max_Drawdown"].std()
    gain_mean, gain_std = data["Max_Gain"].mean(), data["Max_Gain"].std()
    dd_thresh = dd_mean - std_multiplier * dd_std
    dd_thresh = max(dd_thresh, -100)  # Clamp value to no less than -100
    gain_thresh = gain_mean + std_multiplier * gain_std

    # Run backtest
    df_backtest, ar_invest, ar_saving, ar_saving_interest, stopped_early = backtest_weekly_investment(
        df.copy(),
        initial_balance=initial_balance,
        invest_per_week=invest_per_week,
        tp_percent=tp_percent,
        leverage=leverage,
        coeff=coeff,
        std=-dd_thresh,
        start_date=start_date,
        end_date=end_date
    )

    # Plot if needed
    if plots_dir:
        plot_analysis(df, df_backtest, symbol, dd_thresh, gain_thresh, plots_dir, std_multiplier,
                      tp_percent, ar_invest, ar_saving, ar_saving_interest, stopped_early)

    return data, dd_thresh, gain_thresh

def run_all_analyses(
    full_df: pd.DataFrame,
    symbol_list: list,
    std_multiplier: float,
    plots_dir: str,
    tp_percent: float,
    leverage: float,
    coeff_map: dict,
    initial_balance: float,
    invest_per_week: float,
    min_years_required: int
):
    """
    Run analysis for all symbols in the list and return a final summary DataFrame.
    """
    results = {}
    final_summary = []

    for symbol in symbol_list:
        print(f"\n📊 Analyzing {symbol}...")
        df = full_df[full_df["Symbol"] == symbol].copy()
        if df.empty:
            print(f"⚠️ Skipping {symbol}: no data available.")
            continue

        data, dd_thresh, gain_thresh = run_analysis(
            df,
            symbol=symbol,
            std_multiplier=std_multiplier,
            plots_dir=plots_dir,
            tp_percent=tp_percent,
            leverage=leverage,
            coeff=coeff_map.get(symbol, 0.01),
            initial_balance=initial_balance,
            invest_per_week=invest_per_week,
            min_years_required=min_years_required
        )

        if data is None or data.empty:
            continue

        latest_row = df.sort_values("Datetime").iloc[-1]
        latest_date = latest_row["Datetime"]
        latest_price = latest_row["Close"]
        df["Date"] = pd.to_datetime(df["Datetime"])
        cutoff_date = df["Date"].max() - timedelta(days=365 * min_years_required)
        recent_df = df[df["Date"] >= cutoff_date]

        max_price = recent_df["Close"].quantile(0.9)
        min_price = recent_df["Close"].quantile(0.1)
        ar_invest = df["AR_Invest"].iloc[-1] if "AR_Invest" in df.columns else None

        final_summary.append({
            "Symbol": symbol,
            "Date": latest_date,
            "Price": latest_price,
            "Min Price": min_price,
            "Max Price": max_price,
            "Max Gain": gain_thresh,
            "Worst Drawdown": dd_thresh,
            "Coefficient": coeff_map.get(symbol),
            "Annual Return (Simulated)": ar_invest
        })

    # --- Final Summary ---
    final_df = pd.DataFrame(final_summary)

    return final_df