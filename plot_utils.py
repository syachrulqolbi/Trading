import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter

def plot_analysis(df, df_backtest, symbol, dd_thresh, gain_thresh, plots_dir, std_multiplier,
                  tp_percent, ar_invest, ar_saving, ar_saving_interest, stopped_early):
    """
    Generates a multi-panel plot:
    - Max drawdown and gain
    - Weekly investment vs savings growth
    - Close price over time

    Args:
        df (pd.DataFrame): Historical price and drawdown/gain data.
        df_backtest (pd.DataFrame): Backtest investment/savings results.
        symbol (str): Ticker symbol.
        dd_thresh (float): Drawdown threshold line (e.g., -1.96σ).
        gain_thresh (float): Gain threshold line (e.g., +1.96σ).
        plots_dir (str): Directory to save plots.
        std_multiplier (float): Number of standard deviations.
        tp_percent (float): Take profit percent used in backtest.
        ar_invest/saving/saving_interest (float): Annual returns.
        stopped_early (bool): Flag if backtest was stopped early.
    """
    df["Date"] = pd.to_datetime(df["Date"])
    df_backtest["Week"] = pd.to_datetime(df_backtest["Week"], errors='coerce')

    fig = plt.figure(figsize=(14, 10), layout='constrained')
    gs = GridSpec(2, 2, height_ratios=[2, 1], figure=fig)
    fig.suptitle(f"Analysis for {symbol}", fontsize=18, fontweight='bold')

    # ── Top Left: Max Drawdown and Gain ── #
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df["Date"], df["Max_Drawdown"], label="Max Drawdown (%)", linestyle='--', color='crimson')
    ax1.plot(df["Date"], df["Max_Gain"], label="Max Gain (%)", linestyle='-', color='forestgreen')

    ax1.axhline(dd_thresh, color="crimson", linestyle="--", linewidth=1.2,
                label=f"-{std_multiplier}σ Drawdown ({dd_thresh:.2f}%)")
    ax1.axhline(gain_thresh, color="forestgreen", linestyle="--", linewidth=1.2,
                label=f"+{std_multiplier}σ Gain ({gain_thresh:.2f}%)")

    ax1.set_title("Max Drawdown & Gain", fontsize=14)
    ax1.set_ylabel("Return (%)")
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # ── Top Right: Investment vs Savings ── #
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df_backtest['Week'], df_backtest['Cash_Invest'],
             label=f'Investment (TP {tp_percent}% | AR {ar_invest}%)', linestyle='-.')
    ax2.plot(df_backtest['Week'], df_backtest['Cash_Saving'],
             label=f'Saving (AR {ar_saving}%)', linestyle='--')
    ax2.plot(df_backtest['Week'], df_backtest['Cash_Saving_Interest'],
             label=f'Saving + Interest (AR {ar_saving_interest}%)', linestyle=':')

    ax2.set_title(f'{symbol} - Weekly Investment vs Saving', fontsize=14)
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Total Value ($)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.6)

    if stopped_early:
        last_date = df_backtest['Week'].max()
        ax2.axvline(x=last_date, color='red', linestyle='--', linewidth=1.2)
        ax2.text(last_date, ax2.get_ylim()[1] * 0.95, '⚠️ Backtest Stopped Early',
                 color='red', fontsize=9, ha='right', va='top', rotation=90)

    # ── Bottom: Close Price ── #
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(df["Date"], df["Close"], label="Close Price", color="blue", linewidth=1.5)
    ax3.set_title("Close Price Over Time", fontsize=14)
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Price")
    ax3.legend()
    ax3.grid(True, linestyle=':', alpha=0.6)

    # Format x-axis to show years
    ax3.xaxis.set_major_locator(mdates.YearLocator(5))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    ax3.yaxis.set_major_formatter(ScalarFormatter())
    ax3.ticklabel_format(style='plain', axis='y')

    # ── Save Figure ── #
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f"{symbol}_analysis_plot.png"), dpi=300)
    plt.close(fig)


def plot_weighted_normalized_over_time(df: pd.DataFrame, df_final: pd.DataFrame, df_sheet: pd.DataFrame):
    """
    Plots average weighted normalized drawdown/gain over time using action-based weight adjustment.
    """
    def normalize(row):
        action = str(row["Action"]).lower()
        if action == "buy" and pd.notnull(row["Worst Drawdown (%)"]) and row["Worst Drawdown (%)"] != 0:
            return row["Max_Drawdown"] * 100 / -row["Worst Drawdown (%)"]
        elif action == "sell" and pd.notnull(row["Max Gain (%)"]) and row["Max Gain (%)"] != 0:
            return row["Max_Gain"] * 100 / row["Max Gain (%)"]
        return None

    # Add Action column if missing
    if "Action" not in df_final.columns and "Action" in df_sheet.columns:
        df_final = df_final.merge(df_sheet[["Symbol", "Action"]], on="Symbol", how="left")

    # Merge relevant metrics
    df_final = df_final[["Symbol", "Datetime", "Action", "Max_Drawdown", "Max_Gain"]].copy()
    df_info = df_sheet[["Symbol", "Max Gain (%)", "Worst Drawdown (%)"]].drop_duplicates("Symbol")
    df_merged = df_final.merge(df_info, on="Symbol", how="left")
    df_merged["Datetime"] = pd.to_datetime(df_merged["Datetime"])

    # Normalize based on action
    df_merged["Normalized"] = df_merged.apply(normalize, axis=1)
    df_merged.dropna(subset=["Normalized"], inplace=True)

    # Compute weights: halve weight if both buy/sell exist for the same symbol & date
    weight_map = (
        df_merged.groupby(["Symbol", "Datetime"])["Action"]
        .nunique()
        .reset_index(name="Action_Count")
    )
    weight_map["Weight"] = weight_map["Action_Count"].apply(lambda x: 0.5 if x > 1 else 1.0)
    df_merged = df_merged.merge(weight_map[["Symbol", "Datetime", "Weight"]], on=["Symbol", "Datetime"], how="left")
    df_merged["Weighted_Normalized"] = df_merged["Normalized"] * df_merged["Weight"]

    # Average across dates
    avg_per_date = (
        df_merged.groupby("Datetime")["Weighted_Normalized"]
        .mean()
        .reset_index()
        .rename(columns={"Weighted_Normalized": "Avg_Normalized_Value"})
    )

    # ── Plot ── #
    plt.figure(figsize=(12, 6))
    plt.plot(avg_per_date["Datetime"], avg_per_date["Avg_Normalized_Value"], marker='o', label='Avg Normalized Value')
    q5 = avg_per_date["Avg_Normalized_Value"].quantile(0.05)
    plt.axhline(y=q5, color='red', linestyle='--', label=f'5th Percentile ({q5:.2f}%)')
    plt.title("Average Weighted Normalized Drawdown/Gain Over Time")
    plt.xlabel("Date")
    plt.ylabel("Avg Normalized Value (%)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
