import os
import seaborn as sns
import matplotlib.pyplot as plt

PLOTS_DIR = os.path.join(os.getcwd(), "plots")

def cleanup_existing_plots(plot_name: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, plot_name)
    if os.path.exists(plot_path):
        os.remove(plot_path)

def plot_price_gain(data, symbol, avg, std, upper_1std, lower_1std, upper_custom_std, lower_custom_std, std_multiplier):
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))  # Two subplots side by side

    # --- Left Plot: Gain Plot ---
    ax1 = axs[0]
    sns.scatterplot(data=data[data['Price_Gain_Percentage'] >= 0], x='Date', y='Price_Gain_Percentage',
                    label='Gain ≥ 0%', color='green', alpha=0.6, s=10, ax=ax1)
    sns.scatterplot(data=data[data['Price_Gain_Percentage'] < 0], x='Date', y='Price_Gain_Percentage',
                    label='Gain < 0%', color='red', alpha=0.6, s=10, ax=ax1)

    ax1.axhline(avg, color='blue', linestyle='--', label=f'Avg Gain: {avg}%')
    ax1.axhline(upper_1std, color='purple', linestyle='--', label=f'+1 Std: {upper_1std}%')
    ax1.axhline(lower_1std, color='orange', linestyle='--', label=f'-1 Std: {lower_1std}%')
    ax1.axhline(upper_custom_std, color='darkgreen', linestyle='--', label=f'+{std_multiplier} Std: {upper_custom_std}%')
    ax1.axhline(lower_custom_std, color='darkred', linestyle='--', label=f'-{std_multiplier} Std: {lower_custom_std}%')

    ax1.set_title(f'{symbol} - 365-Day Gain %')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('365-Day Gain Percentage (%)')
    ax1.legend(loc='upper center')

    # --- Right Plot: Price Plot ---
    ax2 = axs[1]
    if 'Close' in data.columns:
        sns.lineplot(data=data, x='Date', y='Close', ax=ax2, color='blue', label='Close Price')
        ax2.set_title(f'{symbol} - Closing Price Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No 'Close' price data available", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_title(f'{symbol} - Closing Price Over Time')
        ax2.set_axis_off()

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"{symbol}_gain_price_plot.jpg"), format='jpg', dpi=300)
    plt.close()
