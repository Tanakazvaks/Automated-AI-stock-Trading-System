import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Determine the shared data directory based on OS
if os.name == 'nt':  # Windows
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
else:
    data_dir = "/opt/airflow/data"

# Define input file path
input_csv = os.path.join(data_dir, "DIA_final_optimized_backtest.csv")

# Load backtest results
df = pd.read_csv(input_csv, parse_dates=["timestamp"])

# Ensure PnL column is properly formatted
df["PnL"] = df["PnL"].astype(float)

# **Basic Trade Metrics**
total_trades = df[df["position"] != 0].shape[0]
winning_trades = df[df["PnL"] > 0].shape[0]
losing_trades = total_trades - winning_trades
win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

# Calculate Profit/Loss metrics
total_profit = df[df["PnL"] > 0]["PnL"].sum()
total_loss = abs(df[df["PnL"] < 0]["PnL"].sum())
profit_factor = (total_profit / total_loss) if total_loss > 0 else 0

# Average Win and Loss
avg_win = df[df["PnL"] > 0]["PnL"].mean()
avg_loss = df[df["PnL"] < 0]["PnL"].mean()

# Compute Cumulative PnL and Capital Growth
df["Cumulative_PnL"] = df["PnL"].cumsum()
df["Capital"] = 10000 + df["Cumulative_PnL"]

# ======= Window 1: Profit & Capital Growth =======
fig1, axs1 = plt.subplots(2, 1, figsize=(12, 8))
fig1.suptitle("Trading Strategy Performance", fontsize=16,
              fontweight="bold", color="black")

# Equity Curve - Profit Growth
axs1[0].plot(df["timestamp"], df["Cumulative_PnL"],
             label="Cumulative PnL", color="blue", linewidth=2)
axs1[0].axhline(0, color="red", linestyle="--", alpha=0.7)
axs1[0].set_xlabel("Time")
axs1[0].set_ylabel("Cumulative PnL ($)", fontsize=12, fontweight="bold")
axs1[0].set_title("Equity Curve - Profit Growth",
                  fontsize=14, fontweight="bold")
axs1[0].legend()
axs1[0].grid(True)

# Capital Growth Over Time
axs1[1].plot(df["timestamp"], df["Capital"],
             label="Capital", color="purple", linewidth=2)
axs1[1].axhline(10000, color="gray", linestyle="--", alpha=0.7)
axs1[1].set_xlabel("Time")
axs1[1].set_ylabel("Capital ($)", fontsize=12, fontweight="bold")
axs1[1].set_title("Capital Growth Over Time", fontsize=14, fontweight="bold")
axs1[1].legend()
axs1[1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ======= Window 2: Trade Statistics & Performance Metrics =======
fig2, ax = plt.subplots(figsize=(12, 6))
fig2.suptitle("Trade Performance Metrics", fontsize=16,
              fontweight="bold", color="black")

# Winning vs. Losing Trades Bar Chart
bar_chart = ax.bar(["Winning Trades", "Losing Trades"], [winning_trades, losing_trades],
                   color=["green", "red"], alpha=0.8)
ax.set_ylabel("Number of Trades", fontsize=12, fontweight="bold")
ax.set_title(
    f"Winning vs. Losing Trades (Total: {total_trades})", fontsize=14, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.5)

# Add trade count labels on bars
for bar in bar_chart:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 20, f"{int(height)}",
            ha="center", fontsize=12, fontweight="bold", color="black")

# Display Key Performance Metrics BELOW the chart
fig2.text(0.5, 0.01, f"""
 Average Win: ${avg_win:.2f}      Average Loss: ${avg_loss:.2f}     
 Profit Factor: {profit_factor:.2f}      Win Ratio: {win_rate:.2f}%     
""", wrap=True, horizontalalignment='center',
          fontsize=13, fontweight="bold", color="black",
          bbox=dict(facecolor='whitesmoke', edgecolor='black', boxstyle='round,pad=0.5'))

plt.tight_layout(rect=[0, 0.2, 1, 0.95])
plt.show()
