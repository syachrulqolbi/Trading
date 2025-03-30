import os
import pandas as pd
import matplotlib.pyplot as plt

PLOTS_DIR = os.path.join(os.getcwd(), "plots")

def backtest_weekly_investment(df: pd.DataFrame, initial_balance: float, invest_per_week: float, tp_percent: float,
                                leverage: float, coeff: float, std: float, start_date: str = None, end_date: str = None):
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
                    print(f"🚨 {df['Symbol'].iloc[0] if 'Symbol' in df.columns else ''}: Price dropped below avg trade price - std ({std}) at {row['Week']}. Portfolio wiped out.")
                    cash_invest = 0
                    break

        # Calculate average lot size of current trades
        avg_lot_size = round(sum(list_lot_size) / len(list_lot_size), 4) if list_lot_size else 0.0

        cash_saving += invest_per_week
        weekly_interest_rate = (1 + 0.05) ** (1 / 52) - 1
        cash_saving_interest = (cash_saving_interest + invest_per_week) * (1 + weekly_interest_rate)

        portfolio_history.append({
            "Week": row["Week"], "Close_Price": price, "Profit_TP": round(profit_tp, 2),
            "Cash_Invest": cash_invest, "Cash_Saving": cash_saving,
            "Cash_Saving_Interest": cash_saving_interest,
            "Average_Lot_Size": avg_lot_size
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

    symbol_title = df["Symbol"].iloc[0] if "Symbol" in df.columns else "Symbol"
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_df['Week'], portfolio_df['Cash_Invest'], label=f'Investment (TP {tp_percent}% | AR {ar_invest}%)', linestyle='-.')
    plt.plot(portfolio_df['Week'], portfolio_df['Cash_Saving'], label=f'Saving (AR {ar_saving}%)', linestyle='--')
    plt.plot(portfolio_df['Week'], portfolio_df['Cash_Saving_Interest'], label=f'Saving +5% Interest (AR {ar_saving_interest}%)', linestyle=':')
    plt.title(f'{symbol_title} - Weekly Investment vs Saving')
    plt.xlabel('Week'); plt.ylabel('Total Value ($)')
    plt.legend(); plt.xticks(rotation=45); plt.grid(True); plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"{symbol_title}_investment_plot.jpg"), dpi=300)
    plt.close()

    return portfolio_df, ar_invest, ar_saving, ar_saving_interest
