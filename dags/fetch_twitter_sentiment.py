import praw
import pandas as pd
import os
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load Reddit API Credentials Securely
REDDIT_CLIENT_ID = ""
REDDIT_CLIENT_SECRET = ""
REDDIT_USER_AGENT = ""

# Initialize Reddit API
reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                     client_secret=REDDIT_CLIENT_SECRET,
                     user_agent=REDDIT_USER_AGENT)

# Define search parameters
subreddits = ["cryptocurrency", "stocks", "investing", "stocksandtrading"]
search_terms = ["Dow Jones", "tarriffs",
                "wallstreet", "imports", "trade agreements"]

# Fetch posts
posts = []
for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.new(limit=1000):
        for term in search_terms:
            if term.lower() in post.title.lower() or term.lower() in post.selftext.lower():
                posts.append([
                    datetime.utcfromtimestamp(
                        post.created_utc),
                    post.title,
                    post.selftext,
                    post.score,
                    subreddit_name
                ])

# Convert to DataFrame
df = pd.DataFrame(
    posts, columns=['Timestamp', 'Title', 'Text', 'Upvotes', 'Subreddit'])

# Determine the output directory based on OS
if os.name == 'nt':
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
else:
    data_dir = "/opt/airflow/data"

# Save raw data
raw_filename = os.path.join(data_dir, "reddit_tweets_raw.csv")
df.to_csv(raw_filename, index=False)
print(f"Successfully fetched Reddit posts and saved to '{raw_filename}'!")

# ------------------------------------------------------
# Step 2: Perform Sentiment Analysis using VADER
# ------------------------------------------------------
analyzer = SentimentIntensityAnalyzer()


def get_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']  # Compound score (-1 to 1)


# Apply sentiment analysis (using Title; selftext often empty)
df["Sentiment"] = df["Title"].apply(get_sentiment)

# Save sentiment data
sentiment_filename = os.path.join(data_dir, "reddit_tweets_sentiment.csv")
df.to_csv(sentiment_filename, index=False)
print(f"Sentiment analysis completed & saved to '{sentiment_filename}'!")

# Aggregate sentiment scores by day
df["Date"] = df["Timestamp"].dt.date
daily_sentiment = df.groupby("Date")["Sentiment"].mean()
