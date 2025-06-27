import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
import alpaca_trade_api as tradeapi
from sklearn.preprocessing import LabelEncoder

# Define file paths
data_dir = "/opt/airflow/data" if os.path.exists(
    "/opt/airflow/data") else "data"
input_csv = os.path.join(data_dir, "DIA_updated_strategy.csv")
trade_log_csv = os.path.join(data_dir, "DIA_live_trade_log.csv")

# Alpaca Paper Trading Credentials
ALPACA_API_KEY = ""
ALPACA_SECRET_KEY = ""
BASE_URL = "https://paper-api.alpaca.markets"

# Risk Management Parameters
initial_capital = 10000
risk_per_trade = 0.02
max_drawdown_threshold = -0.15

# ============== Initialize Alpaca API ==============
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY,
                    BASE_URL, api_version='v2')

# ============== Load and Prepare Data ==============
df = pd.read_csv(input_csv, parse_dates=["timestamp"])
df.sort_values("timestamp", inplace=True)

# Fill missing values for relevant features
features = ["RSI_14", "news_sentiment", "reddit_sentiment", "EMA_20", "ATR"]
df[features] = df[features].fillna(method='ffill')

# Encode signals (assuming ml_signal has values like "BUY", "SELL", "HOLD")
label_encoder = LabelEncoder()
df["ml_signal_encoded"] = label_encoder.fit_transform(df["ml_signal"])
# Filter out HOLD signals for trading
df = df[df["ml_signal"] != "HOLD"].copy()

# ============== Functions for Account Info & Risk Management ==============


def get_account_info():
    """Return current Alpaca account details."""
    account = api.get_account()
    return {
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "portfolio_value": float(account.portfolio_value)
    }


def check_drawdown(current_capital, peak_capital):
    """Return True if drawdown exceeds threshold."""
    drawdown = (current_capital - peak_capital) / peak_capital
    return drawdown < max_drawdown_threshold


def close_all_positions():
    """Close all open positions on Alpaca."""
    print(f"{datetime.now()} - Drawdown threshold exceeded. Closing all positions.")
    api.close_all_positions()


def place_bracket_order(symbol, side, qty, risk_to_reward_ratio=2.0):
    """
    Place a market order with bracket (stop loss & take profit).
    For a buy: SL is 2% below, TP is 2% * risk_to_reward_ratio above entry.
    For a sell: SL is 2% above, TP is 2% * risk_to_reward_ratio below entry.
    """
    try:
        last_trade = api.get_latest_trade(symbol)
        entry_price = round(float(last_trade.price), 2)

        if side.lower() == "buy":
            stop_loss_price = round(entry_price * (1 - 0.02), 2)
            take_profit_price = round(
                entry_price * (1 + 0.02 * risk_to_reward_ratio), 2)
        else:  # sell
            stop_loss_price = round(entry_price * (1 + 0.02), 2)
            take_profit_price = round(
                entry_price * (1 - 0.02 * risk_to_reward_ratio), 2)

        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side.lower(),
            type="market",
            time_in_force="gtc",
            order_class="bracket",
            take_profit={"limit_price": take_profit_price},
            stop_loss={"stop_price": stop_loss_price}
        )
        print(f"{datetime.now()} - {side.upper()} {qty} shares of {symbol} @ {entry_price} | SL: {stop_loss_price} | TP: {take_profit_price}")
        return order
    except Exception as e:
        print(f"{datetime.now()} - Order failed for {symbol}: {e}")
        return None


# ============== Paper Trading Execution ==============
symbol = "DIA"
capital = initial_capital
peak_capital = capital
trade_log = []

print(f"{datetime.now()} - Starting paper trading session on Alpaca...")

# Loop through backtested signals (in real trading, signals would come in real-time)
for i, row in df.iterrows():
    # Check current drawdown before placing a new trade
    if check_drawdown(capital, peak_capital):
        close_all_positions()
        break

    # Retrieve signal and price info
    signal = row["ml_signal"]
    close_price = row["close"]

    # Retrieve current account info
    account_info = get_account_info()
    # Determine position size based on buying power and risk per trade
    position_size = max(
        1, int((account_info["buying_power"] * risk_per_trade) / close_price))

    if signal.upper() == "BUY":
        order = place_bracket_order(symbol, "buy", position_size)
    elif signal.upper() == "SELL":
        order = place_bracket_order(symbol, "sell", position_size)
    else:
        continue

    if order is not None:
        trade_log.append({
            "timestamp": datetime.now(),
            "symbol": symbol,
            "side": signal.upper(),
            "qty": position_size,
            "order_id": order.id
        })

    time.sleep(10)

# ============== Save Trade Log ==============
trade_df = pd.DataFrame(trade_log)
trade_df.to_csv(trade_log_csv, index=False)
print(f"{datetime.now()} - Paper trading session completed. Trade log saved to '{trade_log_csv}'.")
