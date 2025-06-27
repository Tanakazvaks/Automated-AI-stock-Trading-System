import requests
import pandas as pd
import os
from transformers import pipeline
from datetime import datetime

# Initialize sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

# Your NewsAPI key
NEWS_API_KEY = ''


def sentiment_to_score(label):
    """Convert sentiment label to numerical score."""
    if label == "POSITIVE":
        return 1
    elif label == "NEGATIVE":
        return -1
    else:
        return 0


def fetch_news_and_analyze_sentiment(query="dow jones", max_results=40):
    """Fetch news articles related to DIA and analyze their sentiment."""
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}'
    response = requests.get(url).json()

    news_data = []
    if 'articles' in response:
        for article in response['articles'][:max_results]:
            title = article['title']
            description = article['description']
            published_at = article['publishedAt']

            # Handle missing description
            if description is None:
                description = ""

            # Run sentiment analysis on title + description
            sentiment_result = sentiment_pipeline(title + ' ' + description)
            sentiment_label = sentiment_result[0]['label']
            sentiment_score = sentiment_to_score(sentiment_label)

            # Convert timestamp and remove timezone
            timestamp = pd.to_datetime(published_at).tz_localize(None)

            news_data.append({
                'timestamp': timestamp,
                'title': title,
                'description': description,
                'sentiment_label': sentiment_label,
                'sentiment_score': sentiment_score
            })

        # Convert to DataFrame
        df = pd.DataFrame(news_data)
        # Extract date to align with daily market data
        df['date'] = df['timestamp'].dt.date

        # Determine output directory based on OS
        if os.name == 'nt':
            data_dir = "data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            csv_filename = f"{data_dir}/news_sentiment.csv"
        else:
            csv_filename = "/opt/airflow/data/news_sentiment.csv"

        # Save the DataFrame to CSV
        df.to_csv(csv_filename, index=False)
        print(f"News sentiment data saved to '{csv_filename}'")
    else:
        print("No news articles found.")


# Fetch & analyze news sentiment
fetch_news_and_analyze_sentiment(query="dow jones", max_results=40)
