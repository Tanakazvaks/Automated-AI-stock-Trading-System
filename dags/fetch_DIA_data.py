import alpaca_trade_api as tradeapi
import pandas as pd
import os
from datetime import datetime, timedelta

# Alpaca API credentials
ALPACA_API_KEY = "PKLHOWJFZY8S0VEH2KF2"
ALPACA_SECRET_KEY = "UCCw4rAGLfef2IBopUCo3QZGkDOV7XV7PZPYJfME"
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    BASE_URL,
    api_version='v2'
)


def fetch_minute_data(ticker="DIA", total_days=30, timeframe="1Min"):
    """
    Fetch intraday (minute-level) data from Alpaca over `total_days` days in 5-day chunks.
    """
    end_datetime = datetime.now() - timedelta(days=2)
    all_data = pd.DataFrame()

    chunk_size = 5
    chunks = (total_days + chunk_size - 1) // chunk_size

    for _ in range(chunks):
        chunk_end = end_datetime
        chunk_start = chunk_end - timedelta(days=chunk_size)

        chunk_end_str = chunk_end.strftime('%Y-%m-%d')
        chunk_start_str = chunk_start.strftime('%Y-%m-%d')

        print(
            f"Fetching {timeframe} data for {ticker} from {chunk_start_str} to {chunk_end_str} ...")
        try:
            bars = api.get_bars(
                symbol=ticker,
                timeframe=timeframe,
                start=chunk_start_str,
                end=chunk_end_str
            ).df

            if not bars.empty:

                bars.index = bars.index.tz_localize(None)

                # Create a 'timestamp' column from the index
                bars['timestamp'] = bars.index.floor("min")

                # Add a 'symbol' column so we can deduplicate by 'symbol' + 'timestamp'
                bars['symbol'] = ticker

                # Reset index fully
                bars.reset_index(drop=True, inplace=True)

                # Append to master DataFrame
                all_data = pd.concat([all_data, bars], ignore_index=True)
                print(
                    f"Pulled {len(bars)} records for {chunk_start_str} to {chunk_end_str}")
            else:
                print(
                    f"No data returned for {chunk_start_str} to {chunk_end_str}")

        except Exception as e:
            print(
                f"Error fetching data for {chunk_start_str} to {chunk_end_str}: {e}")

        # Update end_datetime for next chunk
        end_datetime = chunk_start

    # Drop duplicates and sort by timestamp
    all_data.drop_duplicates(subset=['timestamp', 'symbol'], inplace=True)
    if 'timestamp' in all_data.columns:
        all_data.sort_values('timestamp', inplace=True)

    # Determine the output path based on OS
    if os.name == 'nt':
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        csv_filename = os.path.join(data_dir, f"{ticker}_intraday.csv")
    else:
        csv_filename = f"/opt/airflow/data/{ticker}_intraday.csv"

    all_data.to_csv(csv_filename, index=False)
    print(f" All {len(all_data)} minute bars saved to {csv_filename}")


if __name__ == "__main__":
    fetch_minute_data(ticker="DIA", total_days=30, timeframe="1Min")
