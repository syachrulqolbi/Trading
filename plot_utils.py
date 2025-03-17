import os
import matplotlib.pyplot as plt
import seaborn as sns

PLOTS_DIR = os.path.join(os.getcwd(), "plots")

def cleanup_existing_plots(plot_name: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, plot_name)
    if os.path.exists(plot_path):
        os.remove(plot_path)

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

    plt.xlabel('Date')
    plt.ylabel('365-Day Gain Percentage (%)')
    plt.title(f'{symbol} - 365-Day Price Gain % Over Time')
    plt.legend(loc='upper center')
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"{symbol}_gain_plot.jpg"), format='jpg', dpi=300)
    plt.close()
