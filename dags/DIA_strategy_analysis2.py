import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Determine data directory
data_dir = "data" if os.name == 'nt' else "/opt/airflow/data"
input_csv = os.path.join(data_dir, "DIA_combined_sentiment.csv")
output_csv = os.path.join(data_dir, "DIA_updated_strategy.csv")

# Load Data
df = pd.read_csv(input_csv)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.sort_values("timestamp", inplace=True)

# Feature Engineering


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


df["RSI_14"] = compute_rsi(df["close"])
df["RSI_7"] = compute_rsi(df["close"], period=7)
df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()
df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
df["ATR"] = (df["high"] - df["low"]).rolling(14).mean()

# Scaling sentiment by ATR
df["ATR_scaled"] = df["ATR"] / df["ATR"].max()
df["news_sentiment"] *= df["ATR_scaled"]
df["reddit_sentiment"] *= df["ATR_scaled"]

# Generate trade signals


def generate_trade_signal(row):
    conditions_buy = (row["RSI_14"] < 30) and (row["MACD"] > 0)
    conditions_sell = (row["RSI_14"] > 70) and (row["MACD"] < 0)
    if conditions_buy:
        return 1
    elif conditions_sell:
        return -1
    return 0


df["signal"] = df.apply(generate_trade_signal, axis=1)

# ML data preparation
features = ["RSI_14", "RSI_7", "EMA_20", "EMA_50", "MACD",
            "ATR_scaled", "news_sentiment", "reddit_sentiment"]
df_ml = df.dropna(subset=features + ["signal"]).copy()

# Label encoding
label_mapping = {-1: 0, 0: 1, 1: 2}
inverse_label_mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}
df_ml.loc[:, "signal"] = df_ml["signal"].map(label_mapping)

X, y = df_ml[features], df_ml["signal"]

# Cross-validation
tscv = TimeSeriesSplit(n_splits=5)
model = xgb.XGBClassifier(
    objective='multi:softmax',
    eval_metric='mlogloss',
    max_depth=3,
    learning_rate=0.01,
    n_estimators=200,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42
)

scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
print(f"Cross-validation Accuracy: {np.mean(scores):.4f}")

# Train final model
model.fit(X, y)
df["ml_signal"] = model.predict(df[features].fillna(method='bfill'))
df["ml_signal"] = df["ml_signal"].map(inverse_label_mapping).fillna("HOLD")

# Save results
df.to_csv(output_csv, index=False)
print(f" ML-Enhanced Strategy Saved as '{output_csv}'")
print(df["ml_signal"].value_counts())
