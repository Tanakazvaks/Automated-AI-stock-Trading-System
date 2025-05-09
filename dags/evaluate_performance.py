import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Determine the shared data directory based on OS
if os.name == 'nt':
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
else:
    data_dir = "/opt/airflow/data"

# Define input and output file paths
input_csv = os.path.join(data_dir, "DIA_final_optimized_backtest.csv")
performance_csv = os.path.join(data_dir, "DIA_performance_results.csv")

# Load backtest results
df = pd.read_csv(input_csv, parse_dates=["timestamp"])

# Ensure PnL column is properly formatted
df["PnL"] = df["PnL"].astype(float)

# **Basic Trade Metrics**
total_trades = df[df["position"] != 0].shape[0]
winning_trades = df[df["PnL"] > 0].shape[0]
losing_trades = total_trades - winning_trades
win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
avg_trade_return = df["PnL"].mean()

# **Profit & Loss Metrics**
total_profit = df[df["PnL"] > 0]["PnL"].sum()
total_loss = abs(df[df["PnL"] < 0]["PnL"].sum())  # Absolute loss
profit_factor = (total_profit / total_loss) if total_loss > 0 else np.nan

# **Max Drawdown Calculation**
df["Cumulative_PnL"] = df["PnL"].cumsum()
rolling_max = df["Cumulative_PnL"].cummax()
drawdown = rolling_max - df["Cumulative_PnL"]
max_drawdown = drawdown.max()

# **Sharpe Ratio Calculation (Fixed)**
risk_free_rate = 0.01

# **Ensure returns are non-zero before calculating Sharpe Ratio**
df["returns"] = df["PnL"].replace(0, np.nan).pct_change().dropna()
if df["returns"].empty or df["returns"].std() == 0:
    sharpe_ratio = 0
else:
    sharpe_ratio = (df["returns"].mean() - risk_free_rate) / \
        df["returns"].std()

# **Cumulative Return Calculation**
initial_capital = 10000
final_capital = initial_capital + df["PnL"].sum()
cumulative_return = ((final_capital - initial_capital) / initial_capital) * 100

# **Performance Summary**
print("\n**Trading Strategy Performance Evaluation**")
print(f"Total Trades: {total_trades}")
print(f"Winning Trades: {winning_trades}")
print(f"Losing Trades: {losing_trades}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Average Trade Return: ${avg_trade_return:.2f}")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Max Drawdown: ${max_drawdown:.2f}")
print(f"Cumulative Return: {cumulative_return:.2f}%")

# **Save Results to CSV**
performance_results = pd.DataFrame({
    "Metric": [
        "Total Trades", "Winning Trades", "Losing Trades",
        "Win Rate (%)", "Average Trade Return ($)", "Profit Factor",
        "Max Drawdown ($)", "Sharpe Ratio", "Cumulative Return (%)"
    ],
    "Value": [
        total_trades, winning_trades, losing_trades,
        round(win_rate, 2), round(
            avg_trade_return, 2), round(profit_factor, 2),
        round(max_drawdown, 2), round(
            sharpe_ratio, 2), round(cumulative_return, 2)
    ]
})
performance_results.to_csv(performance_csv, index=False)
print("\n Performance results saved as '{}'.".format(performance_csv))

# **Visualizing Equity Curve & Drawdown**
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# **Equity Curve**
axs[0].plot(df["timestamp"], df["Cumulative_PnL"],
            label="Equity Curve", color="blue")
axs[0].axhline(0, color="red", linestyle="--", alpha=0.7)
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Cumulative PnL ($)")
axs[0].set_title("Equity Curve - Strategy Performance")
axs[0].legend()
axs[0].grid(True)

# **Drawdown Plot**
axs[1].plot(df["timestamp"], drawdown,
            label="Drawdown", color="red", alpha=0.6)
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Drawdown ($)")
axs[1].set_title("Maximum Drawdown Over Time")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
