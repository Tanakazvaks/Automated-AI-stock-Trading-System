# Trading System Automation with Docker and Apache Airflow

## Overview

This project automates a trading system using containerized Python scripts and Apache Airflow for scheduling and orchestration. The system includes modules for:
- **Market Data Fetching:** Uses the Alpaca API to pull minute-level market data.
- **Sentiment Analysis:** Analyzes news and social media sentiment using NewsAPI, Reddit, and Hugging Face Transformers.
- **Data Merging & Strategy Analysis:** Combines market data with sentiment scores and runs trading strategy analyses.
- **Backtesting & Performance Evaluation:** Evaluates the trading strategy using historical data.
- **Live Trading Execution:** Executes live trades via the Alpaca API.

All components are containerized with Docker, and Airflow is used to schedule and manage the pipeline.


## Project Structure

├── dags │ ├── backtest_intraday.py │ ├── evaluate_performance.py │ ├── fetch_news_sentiment.py │ ├── fetch_DIA_data.py │ ├── fetch_twitter_sentiment.py │ ├── merge_market_sentiment.py │ ├── DIA_strategy_analysis.py │ ├── trade_with_alpaca.py │ └── trading_pipeline_dag.py ├── data │ └── [Generated CSV files will be stored here] ├── logs │ └── [Airflow logs] ├── Dockerfile.airflow ├── docker-compose.yml └── README.md


## Prerequisites

- **Docker** and **Docker Compose** installed on your machine.
- **Python 3.10** (or later) installed if you plan to test scripts locally.
- Valid API credentials for:
  - **Alpaca:** For market data and live trading.
  - **NewsAPI:** For fetching news articles.
  - **Reddit** For fetching twitts
- Basic familiarity with the command line.

## Setup and Installation

### 1. Clone the Repository

*1. Clone this repository to your local machine:

```bash
git clone https://your-repository-url.git
cd your-repository-folder


*2. Configure API Credentials
-Open the relevant scripts (e.g., fetch_DIA_data.py, fetch_news_sentiment.py, trade_with_alpaca.py) and update them with your API keys.

-In docker-compose.yml, replace YOUR_FERNET_KEY_HERE and YOUR_SECRET_KEY_HERE with strong, random keys. Generate them using this Python snippet:

import os, base64
print("Fernet Key:", base64.urlsafe_b64encode(os.urandom(32)).decode())
print("Secret Key:", os.urandom(24).hex())


*3. Build the Custom Airflow Image
The project uses a custom Airflow image that installs all required Python dependencies. To build the image, run:

docker compose build


*4. Initialize Airflow
Initialize the Airflow metadata database and create the admin user by running:

docker compose up airflow-init

After initialization completes (you should see "Initialization done"), press Ctrl+C to exit the logs.

*5. Start the Airflow Services
Launch the Airflow services (webserver, scheduler, and PostgreSQL) in detached mode:

docker compose up -d


The Airflow webserver will be accessible at http://localhost:8080.


*6. Verify the DAGs

-Open your browser and go to http://localhost:8080.
-Log in using the credentials (default: admin / admin).
-Verify that the DAG trading_pipeline appears in the DAG list.
-Unpause the DAG and click the Trigger DAG button to run the pipeline manually.
-Monitor task progress using the Graph View or Tree View. Click on tasks to view detailed logs.

*Shared Data Folder
All data files (CSV outputs, logs, etc.) are stored in a shared folder:

-On Windows: The local data folder.

-On Linux (inside Airflow containers): /opt/airflow/data

Make sure this folder exists in your project directory; it is mounted in the Docker Compose file.


*7. Trigger and Monitor the Pipeline
Manual Trigger:
-Unpause the DAG if it’s paused, then click the trigger (play) button to run it manually.

Monitor Execution:
-Use the Airflow UI’s Graph or Tree View to check task statuses and click individual tasks to view their logs.

**Running the Program Locally for Testing**

  If you wish to test individual scripts outside of Airflow:

  1. Activate your virtual environment (if using one):

     .\venv\Scripts\Activate.ps1


  2. Run a script manually:

     python fetch_DIA_data.py

     Ensure you have the required packages installed locally.

**Troubleshooting**

-File Not Found Errors:
      Ensure that all scripts read from and write to the shared data folder. The Docker Compose file mounts the local data folder to /opt/airflow/data inside containers.

-API Errors:
      Verify your API credentials and subscription levels.

-Dependency Issues:
      If a package is missing, update Dockerfile.airflow with the required dependency and rebuild the image.

-Log Inspection:
      Use the Airflow UI to inspect task logs and diagnose any issues.


## Running Tests

This project includes a test suite to ensure that all components of the trading system work as expected. Tests are implemented using Python's built-in `unittest` framework.

### Running Tests Locally

To run the test suite on your local machine:

1. Ensure you have all dependencies installed:
   ```bash
   pip install -r requirements.txt
