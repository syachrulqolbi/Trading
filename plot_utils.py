import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter

def plot_analysis(df, df_backtest, symbol, dd_thresh, gain_thresh, plots_dir, std_multiplier,
                  tp_percent, ar_invest, ar_saving, ar_saving_interest, stopped_early):
    """
    Plots drawdown, gain, close price, and investment comparison over time.
    """
    df["Date"] = pd.to_datetime(df["Date"])

    fig = plt.figure(figsize=(14, 10), layout='constrained')
    gs = GridSpec(2, 2, height_ratios=[2, 1], figure=fig)
    
    fig.suptitle(f"Analysis for {symbol}", fontsize=18, fontweight='bold')

    # --- Top Left: Max Drawdown and Gain --- #
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

    # --- Top Right: Investment vs Saving --- #
    df_backtest["Week"] = pd.to_datetime(df_backtest["Week"], errors='coerce')  # Ensure datetime

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df_backtest['Week'], df_backtest['Cash_Invest'],
            label=f'Investment (TP {tp_percent}% | AR {ar_invest}%)', linestyle='-.')
    ax2.plot(df_backtest['Week'], df_backtest['Cash_Saving'],
            label=f'Saving (AR {ar_saving}%)', linestyle='--')
    ax2.plot(df_backtest['Week'], df_backtest['Cash_Saving_Interest'],
            label=f'Saving +5% Interest (AR {ar_saving_interest}%)', linestyle=':')
    ax2.set_title(f'{symbol} - Weekly Investment vs Saving', fontsize=14)
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Total Value ($)')
    ax2.legend(fontsize=8)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, linestyle=':', alpha=0.6)
    if stopped_early:
        last_date = df_backtest['Week'].max()
        ax2.axvline(x=last_date, color='red', linestyle='--', linewidth=1.2)
        ax2.text(last_date, ax2.get_ylim()[1] * 0.95, '⚠️ Backtest Stopped Early',
                 color='red', fontsize=9, ha='right', va='top', rotation=90)

    # --- Bottom Full Width: Close Price --- #
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(df["Date"], df["Close"], label="Close Price", color="blue", linewidth=1.5)
    ax3.set_title("Close Price Over Time", fontsize=14)
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Price")
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend()

    ax3.xaxis.set_major_locator(mdates.YearLocator(5))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    ax3.yaxis.set_major_formatter(ScalarFormatter())
    ax3.ticklabel_format(style='plain', axis='y')

    # Save
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"{symbol}_analysis_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)