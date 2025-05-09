import os
import pandas as pd

# Determine the shared data directory based on OS
if os.name == 'nt':
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
else:
    data_dir = "/opt/airflow/data"

# Define file paths
market_file = os.path.join(data_dir, "DIA_intraday.csv")
news_file = os.path.join(data_dir, "news_sentiment.csv")
reddit_file = os.path.join(data_dir, "reddit_tweets_sentiment.csv")
output_file = os.path.join(data_dir, "DIA_combined_sentiment.csv")

# Load Market Data
df_market = pd.read_csv(market_file)
df_market["timestamp"] = pd.to_datetime(df_market["timestamp"])
df_market["date"] = df_market["timestamp"].dt.date

# Load News Sentiment Data
df_news = pd.read_csv(news_file)
df_news["timestamp"] = pd.to_datetime(df_news["timestamp"])
df_news["date"] = df_news["timestamp"].dt.date

# Load Reddit Sentiment Data
df_reddit = pd.read_csv(reddit_file)
df_reddit["timestamp"] = pd.to_datetime(df_reddit["Timestamp"])
df_reddit["date"] = df_reddit["timestamp"].dt.date

# Aggregate Daily Sentiment Scores
df_news_daily = df_news.groupby("date")["sentiment_score"].mean().reset_index()
df_reddit_daily = df_reddit.groupby("date")["Sentiment"].mean().reset_index()

# Normalize Sentiment Scores
df_news_daily["sentiment_score"] = (df_news_daily["sentiment_score"] -
                                    df_news_daily["sentiment_score"].mean()) / df_news_daily["sentiment_score"].std()
df_reddit_daily["Sentiment"] = (df_reddit_daily["Sentiment"] -
                                df_reddit_daily["Sentiment"].mean()) / df_reddit_daily["Sentiment"].std()

# Merge Market Data with Sentiments
df_combined = df_market.merge(df_news_daily, on="date", how="left")
df_combined = df_combined.merge(df_reddit_daily, on="date", how="left")

# Rename Sentiment Columns for Clarity
df_combined.rename(columns={"sentiment_score": "news_sentiment",
                   "Sentiment": "reddit_sentiment"}, inplace=True)

# Fill Missing Sentiment Values with 0
df_combined["news_sentiment"].fillna(0, inplace=True)
df_combined["reddit_sentiment"].fillna(0, inplace=True)

# Save the Final Combined Data
df_combined.to_csv(output_file, index=False)

print("✅ Merging Complete! Data saved as '{}'.".format(output_file))
print("\n--- Sample Data After Merging ---")
print(
    df_combined[["timestamp", "news_sentiment", "reddit_sentiment"]].head(10))
