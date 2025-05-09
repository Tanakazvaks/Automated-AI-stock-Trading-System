from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'retries': 0,
}

with DAG(
    'trading_pipeline',
    default_args=default_args,
    description='Trading pipeline: fetch data, analysis, backtest, visualization and live trading',
    schedule_interval='@daily',
    start_date=datetime(2025, 3, 1),
    catchup=False,
) as dag:

    fetch_DIA_data = BashOperator(
        task_id='fetch_DIA_data',
        bash_command='python /opt/airflow/dags/fetch_DIA_data.py'
    )

    fetch_news_sentiment = BashOperator(
        task_id='fetch_news_sentiment',
        bash_command='python /opt/airflow/dags/fetch_news_sentiment.py'
    )

    fetch_twitter_sentiment = BashOperator(
        task_id='fetch_twitter_sentiment',
        bash_command='python /opt/airflow/dags/fetch_twitter_sentiment.py'
    )

    merge_market_sentiment = BashOperator(
        task_id='merge_market_sentiment',
        bash_command='python /opt/airflow/dags/merge_market_sentiment.py'
    )

    DIA_strategy_analysis = BashOperator(
        task_id='DIA_strategy_analysis',
        # <-- Make sure you rename the file if needed
        bash_command='python /opt/airflow/dags/DIA_strategy_analysis.py'
    )

    backtest_intraday = BashOperator(
        task_id='backtest_intraday',
        bash_command='python /opt/airflow/dags/backtest_intraday.py'
    )

    evaluate_performance = BashOperator(
        task_id='evaluate_performance',
        bash_command='python /opt/airflow/dags/evaluate_performance.py'
    )

    trade_with_alpaca = BashOperator(
        task_id='trade_with_alpaca',
        bash_command='python /opt/airflow/dags/trade_with_alpaca.py'
    )

    visualize_backtest = BashOperator(
        task_id='visualize_backtest',
        bash_command='python /opt/airflow/dags/visualize_backtest.py'
    )

    # Define the sequential order of tasks:
    fetch_DIA_data >> fetch_news_sentiment >> fetch_twitter_sentiment >> merge_market_sentiment >> DIA_strategy_analysis >> backtest_intraday >> evaluate_performance >> visualize_backtest >> trade_with_alpaca
