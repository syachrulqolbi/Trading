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

def plot_average_standardized_drawdown(df_drawdown_avg, th_drawdown, plots_dir):
    plt.figure(figsize=(14, 7))

    # Convert values to negative
    df_drawdown_avg['Avg_Standardized_Drawdown'] = -abs(df_drawdown_avg['Avg_Standardized_Drawdown'])
    th_drawdown = -abs(th_drawdown)

    plt.plot(df_drawdown_avg['Date'], df_drawdown_avg['Avg_Standardized_Drawdown'], linestyle='-', linewidth=1.5, label='Avg Standardized Drawdown')

    # Plotting Quantile Line
    plt.axhline(y=th_drawdown, color='red', linestyle='--', linewidth=1.2, 
                label=f"0.9 Quantile: {th_drawdown:.2f}%")

    plt.title("Average Standardized Max Drawdown by Date")
    plt.xlabel("Date")
    plt.ylabel("Avg Standardized Max Drawdown (%)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # ── Save Figure ── #
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "analysis_max_drawdown_overtime_plot.png"), dpi=300)
    plt.show()