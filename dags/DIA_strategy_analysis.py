import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
<<<<<<< HEAD
import matplotlib.pyplot as plt
=======
>>>>>>> d1afe7f098ec4158d100c2f235bd7854037a9f89

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DESIRED_PRECISION = 0.80   # target precision for BUY/SELL
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ─── PATHS ─────────────────────────────────────────────────────────────────────
data_dir = "data" if os.name == 'nt' else "/opt/airflow/data"
input_csv = os.path.join(data_dir, "DIA_combined_sentiment.csv")
output_csv = os.path.join(data_dir, "DIA_updated_strategy.csv")

# ─── LOAD & FEATURES ────────────────────────────────────────────────────────────
df = pd.read_csv(input_csv, parse_dates=["timestamp"])
df.sort_values("timestamp", inplace=True)
df = df.ffill().bfill()


def compute_rsi(s, period=14):
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


df["RSI_14"] = compute_rsi(df["close"])
df["RSI_7"] = compute_rsi(df["close"], period=7)
df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()
df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
df["MACD"] = df["EMA_20"] - df["EMA_50"]
df["ATR"] = (df["high"] - df["low"]).rolling(14).mean().ffill().bfill()
df["ATR_scaled"] = df["ATR"] / df["ATR"].max()
df["news_sentiment"] *= df["ATR_scaled"]
df["reddit_sentiment"] *= df["ATR_scaled"]

# ─── RULE-BASED LABELS ─────────────────────────────────────────────────────────


def generate_trade_signal(r):
    buy = (
        (r.RSI_14 < 30) &
        (r.MACD > 0) &
        (r.ATR_scaled > 0.2) &
        ((r.news_sentiment > 0) | (r.reddit_sentiment > 0))
    )
    sell = (
        (r.RSI_14 > 70) &
        (r.MACD < 0) &
        (r.ATR_scaled > 0.2) &
        ((r.news_sentiment < 0) | (r.reddit_sentiment < 0))
    )
    return 1 if buy else (-1 if sell else 0)


df["signal"] = df.apply(generate_trade_signal, axis=1)
if df["signal"].nunique() < 2:
<<<<<<< HEAD
    df["signal"] = np.sign(df["close"].shift(-1) -
                           df["close"]).ffill().fillna(0).astype(int)
=======
    df["signal"] = (
        np.sign(df["close"].shift(-1) - df["close"])
          .ffill()
          .fillna(0)
          .astype(int)
    )
>>>>>>> d1afe7f098ec4158d100c2f235bd7854037a9f89

# ─── PREPARE ML DATA ────────────────────────────────────────────────────────────
features = [
    "RSI_14", "RSI_7", "EMA_20", "EMA_50",
    "MACD", "ATR_scaled", "news_sentiment", "reddit_sentiment"
]
df_ml = df.dropna(subset=features + ["signal"]).copy()
df_ml["label"] = df_ml["signal"].map({-1: 0, 0: 1, 1: 2})

X_all, y_all = df_ml[features], df_ml["label"]

# ─── SPLIT & BALANCE ───────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_all
)
df_tr = pd.concat([X_train, y_train.rename("label")], axis=1)
hold = df_tr[df_tr.label == 1]
buy = df_tr[df_tr.label == 2]
sell = df_tr[df_tr.label == 0]
n_trades = len(buy) + len(sell)
<<<<<<< HEAD
hold_ds = resample(hold, replace=False, n_samples=n_trades,
                   random_state=RANDOM_STATE)
df_bal = pd.concat([buy, sell, hold_ds])
X_tr = df_bal[features]
y_tr = df_bal["label"]
X_te = X_test
y_te = y_test
=======
hold_ds = resample(
    hold,
    replace=False,
    n_samples=n_trades,
    random_state=RANDOM_STATE
)
df_bal = pd.concat([buy, sell, hold_ds])
X_tr, y_tr = df_bal[features], df_bal["label"]
X_te, y_te = X_test, y_test
>>>>>>> d1afe7f098ec4158d100c2f235bd7854037a9f89

# ─── TRAIN ─────────────────────────────────────────────────────────────────────
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="mlogloss",
    max_depth=3,
    learning_rate=0.01,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.5,
    gamma=0.5,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=RANDOM_STATE
)
model.fit(X_tr, y_tr)

# ─── THRESHOLD SELECTION ───────────────────────────────────────────────────────
probs_te = model.predict_proba(X_te)

<<<<<<< HEAD
# BUY threshold: smallest threshold with precision ≥ DESIRED_PRECISION
prec_b, rec_b, thr_b = precision_recall_curve(
    (y_te == 2).astype(int), probs_te[:, 2])
valid_b = [t for p, t in zip(prec_b[1:], thr_b) if p >= DESIRED_PRECISION]
th_buy = min(valid_b) if valid_b else 0.5

# SELL threshold
prec_s, rec_s, thr_s = precision_recall_curve(
    (y_te == 0).astype(int), probs_te[:, 0])
valid_s = [t for p, t in zip(prec_s[1:], thr_s) if p >= DESIRED_PRECISION]
th_sell = min(valid_s) if valid_s else 0.5

print(f"Chosen BUY threshold:  {th_buy:.3f}")
print(f"Chosen SELL threshold: {th_sell:.3f}")

# ─── EVALUATION ────────────────────────────────────────────────────────────────
y_pred_raw = model.predict(X_te)
print("\nRaw performance:")
print(classification_report(
    y_te, y_pred_raw,
    labels=[0, 1, 2],
    target_names=["SELL", "HOLD", "BUY"],
    zero_division=0
))

y_pred_thr = []
for p in model.predict_proba(X_te):
    if p[2] > th_buy:
        y_pred_thr.append(2)
    elif p[0] > th_sell:
        y_pred_thr.append(0)
    else:
        y_pred_thr.append(1)

print("\nThresholded performance:")
print(classification_report(
    y_te, y_pred_thr,
    labels=[0, 1, 2],
    target_names=["SELL", "HOLD", "BUY"],
    zero_division=0
))

# ─── FINAL SIGNALS & SAVE ─────────────────────────────────────────────────────
probs_all = model.predict_proba(df_ml[features].ffill())
ml_sig = []
for p in probs_all:
    if p[2] > th_buy:
        ml_sig.append("BUY")
    elif p[0] > th_sell:
        ml_sig.append("SELL")
    else:
        ml_sig.append("HOLD")

df.loc[df_ml.index, "ml_signal"] = ml_sig
=======
prec_b, rec_b, thr_b = precision_recall_curve(
    (y_te == 2).astype(int), probs_te[:, 2]
)
valid_b = [t for p, t in zip(prec_b[1:], thr_b) if p >= DESIRED_PRECISION]
th_buy = min(valid_b) if valid_b else 0.5

prec_s, rec_s, thr_s = precision_recall_curve(
    (y_te == 0).astype(int), probs_te[:, 0]
)
valid_s = [t for p, t in zip(prec_s[1:], thr_s) if p >= DESIRED_PRECISION]
th_sell = min(valid_s) if valid_s else 0.5

print(f"Chosen BUY threshold:  {th_buy:.3f}")
print(f"Chosen SELL threshold: {th_sell:.3f}")

# ─── EVALUATION ────────────────────────────────────────────────────────────────
y_pred_raw = model.predict(X_te)
print("\nRaw performance:")
print(classification_report(
    y_te, y_pred_raw,
    labels=[0, 1, 2],
    target_names=["SELL", "HOLD", "BUY"],
    zero_division=0
))

y_pred_thr = []
for p in probs_te:
    if p[2] > th_buy:
        y_pred_thr.append(2)
    elif p[0] > th_sell:
        y_pred_thr.append(0)
    else:
        y_pred_thr.append(1)

print("\nThresholded performance:")
print(classification_report(
    y_te, y_pred_thr,
    labels=[0, 1, 2],
    target_names=["SELL", "HOLD", "BUY"],
    zero_division=0
))

# ─── FINAL SIGNALS & SAVE ─────────────────────────────────────────────────────
df["ml_signal"] = "HOLD"

probs_all = model.predict_proba(df_ml[features].ffill())
ml_sig = []
for p in probs_all:
    if p[2] > th_buy:
        ml_sig.append("BUY")
    elif p[0] > th_sell:
        ml_sig.append("SELL")
    else:
        ml_sig.append("HOLD")

df.loc[df_ml.index, "ml_signal"] = ml_sig

>>>>>>> d1afe7f098ec4158d100c2f235bd7854037a9f89
df.to_csv(output_csv, index=False)
print(f"\nSaved tuned strategy to '{output_csv}'")
