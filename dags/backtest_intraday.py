import os
import pandas as pd
import numpy as np

# Load Data
data_dir = "data" if os.name == 'nt' else "/opt/airflow/data"
input_csv = os.path.join(data_dir, "DIA_updated_strategy.csv")
output_csv = os.path.join(data_dir, "DIA_final_optimized_backtest.csv")

df = pd.read_csv(input_csv, parse_dates=["timestamp"])
df.sort_values("timestamp", inplace=True)

df = df.ffill().copy()
df["ATR"] = df["ATR"].replace(0, np.nan).ffill().bfill()

# Risk Management
initial_capital = 10000
risk_per_trade = 0.01
capital = initial_capital
max_drawdown_threshold = -0.20
peak_equity = capital
drawdown_exceeded = False

df["position"] = 0
df["PnL"] = 0.0

for i in range(len(df) - 5):
    if drawdown_exceeded:
        break

    signal = df.iloc[i]["ml_signal"]
    entry_price = df.iloc[i]["close"]
    atr = df.iloc[i]["ATR"]

    if np.isnan(atr) or atr <= 0:
        continue

    position_size = int((capital * risk_per_trade) / atr)
    if position_size <= 0:
        continue

    target_multiplier = 2
    stop_multiplier = 1

    if signal == "BUY":
        target = entry_price + target_multiplier * atr
        stop_loss = entry_price - stop_multiplier * atr
        future_high = df.iloc[i+1:i+6]["high"].max()
        future_low = df.iloc[i+1:i+6]["low"].min()

        if future_high >= target:
            pnl = (target - entry_price) * position_size
        elif future_low <= stop_loss:
            pnl = (stop_loss - entry_price) * position_size
        else:
            pnl = (df.iloc[i+5]["close"] - entry_price) * position_size
        df.at[i, "position"] = 1

    elif signal == "SELL":
        target = entry_price - target_multiplier * atr
        stop_loss = entry_price + stop_multiplier * atr
        future_high = df.iloc[i+1:i+6]["high"].max()
        future_low = df.iloc[i+1:i+6]["low"].min()

        if future_low <= target:
            pnl = (entry_price - target) * position_size
        elif future_high >= stop_loss:
            pnl = (entry_price - stop_loss) * position_size
        else:
            pnl = (entry_price - df.iloc[i+5]["close"]) * position_size
        df.at[i, "position"] = -1

    else:
        pnl = 0

    capital += pnl
    df.at[i, "PnL"] = pnl
    peak_equity = max(peak_equity, capital)
    drawdown = (capital - peak_equity) / peak_equity

    if drawdown < max_drawdown_threshold:
        drawdown_exceeded = True

# Final results
final_pnl = capital - initial_capital
win_trades = len(df[df["PnL"] > 0])
loss_trades = len(df[df["PnL"] < 0])
total_trades = win_trades + loss_trades
win_rate = win_trades / total_trades * 100 if total_trades else 0
cum_return = final_pnl / initial_capital * 100

df.to_csv(output_csv, index=False)

print("\n Optimized Backtest Complete")
print(f"Total Trades: {total_trades}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Total PnL: ${final_pnl:.2f}")
print(f"Cumulative Return: {cum_return:.2f}%")
