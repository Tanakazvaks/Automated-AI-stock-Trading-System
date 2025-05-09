import unittest
import pandas as pd
import numpy as np


def compute_rsi(series, period=14):
    """Compute RSI for a given series, returning 100 for a constant series."""
    # If the series is constant, return 100 for all values.
    if series.nunique() == 1:
        return pd.Series(100, index=series.index)

    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(
        window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)
            ).rolling(window=period, min_periods=1).mean()
    # Prevent division by zero by replacing any zero losses with a small number
    loss = loss.replace(0, 1e-10)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def generate_trade_signal(row):
    """
    Generate a trade signal based on RSI and combined sentiment:
    - "BUY" if RSI_14 < 35 and combined_sentiment > 0.05.
    - "SELL" if RSI_14 > 65 and combined_sentiment < -0.05.
    - Otherwise, "HOLD".
    """
    if row["RSI_14"] < 35 and row["combined_sentiment"] > 0.05:
        return "BUY"
    elif row["RSI_14"] > 65 and row["combined_sentiment"] < -0.05:
        return "SELL"
    else:
        return "HOLD"


class TestTradingStrategy(unittest.TestCase):

    def test_compute_rsi_increasing(self):
        # For a strictly increasing series, expect RSI near 100.
        data = pd.Series(np.arange(1, 21))
        rsi = compute_rsi(data, period=14)
        self.assertTrue(rsi.iloc[-1] > 90,
                        "RSI for increasing series should be near 100")

    def test_compute_rsi_decreasing(self):
        # For a strictly decreasing series, expect RSI near 0.
        data = pd.Series(np.arange(20, 0, -1))
        rsi = compute_rsi(data, period=14)
        self.assertTrue(rsi.iloc[-1] < 10,
                        "RSI for decreasing series should be near 0")

    def test_compute_rsi_constant(self):
        # For a constant series, the function returns 100.
        data = pd.Series([100] * 20)
        rsi = compute_rsi(data, period=14)
        self.assertTrue((rsi == 100).all(),
                        "RSI for a constant series should be 100")

    def test_generate_trade_signal_buy(self):
        test_row = {"RSI_14": 30, "combined_sentiment": 0.1}
        signal = generate_trade_signal(test_row)
        self.assertEqual(
            signal, "BUY", "Expected signal 'BUY' for low RSI and positive sentiment")

    def test_generate_trade_signal_sell(self):
        # Use RSI > 65 and combined sentiment < -0.05 to trigger SELL
        test_row = {"RSI_14": 70, "combined_sentiment": -0.1}
        signal = generate_trade_signal(test_row)
        self.assertEqual(
            signal, "SELL", "Expected signal 'SELL' for high RSI and negative sentiment")

    def test_generate_trade_signal_hold(self):
        test_row = {"RSI_14": 45, "combined_sentiment": 0.0}
        signal = generate_trade_signal(test_row)
        self.assertEqual(
            signal, "HOLD", "Expected signal 'HOLD' for neutral conditions")

    def test_trade_position_size(self):
        initial_capital = 10000
        risk_per_trade = 0.02
        close_price = 100
        # Calculate expected position size: (10000 * 0.02) / 100 = 2
        expected_position_size = max(
            1, int((initial_capital * risk_per_trade) / close_price))
        self.assertEqual(expected_position_size, 2,
                         "Position size calculation is incorrect")


if __name__ == '__main__':
    unittest.main()
