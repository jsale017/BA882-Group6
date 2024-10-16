from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import logging
import requests
import json
from google.cloud import storage, secretmanager, bigquery
import pandas as pd

# Helper function to get API key from Secret Manager
def get_alphavantage_api_key():
    client = secretmanager.SecretManagerServiceClient()
    secret_name = "projects/finnhub-pipeline-ba882/secrets/alphavantage-api-key/versions/latest"
    response = client.access_secret_version(request={"name": secret_name})
    return response.payload.data.decode("UTF-8")

# Helper function to upload raw data to GCS
def upload_to_gcs(bucket_name, file_name, data):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(data)
    logging.info(f"Uploaded {file_name} to bucket {bucket_name}")

# Download from GCS
def download_from_gcs(bucket_name, file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    raw_data = blob.download_as_string()
    logging.info(f"Downloaded {file_name} from bucket {bucket_name}")
    return json.loads(raw_data)

# Parse stock data
def parse_stock_data(raw_data):
    try:
        time_series = raw_data.get("Time Series (Daily)", {})
        parsed_data = []
        for date, daily_data in time_series.items():
            parsed_record = {
                "symbol": symbol,
                "date": date,
                "open": daily_data.get("1. open"),
                "high": daily_data.get("2. high"),
                "low": daily_data.get("3. low"),
                "close": daily_data.get("4. close"),
                "volume": daily_data.get("5. volume"),
            }
            parsed_data.append(parsed_record)
        logging.info("Successfully parsed stock data")
        return parsed_data
    except KeyError as e:
        logging.error(f"KeyError during parsing: {str(e)}")
        return None

# Extract task logic
def extract_data():
    logging.info("Starting data extraction")
    try:
        api_key = get_alphavantage_api_key()
        stock_symbols = ['AAPL', 'NFLX', 'MSFT', 'NVDA', 'AMZN']
        bucket_name = 'finnhub-financial-data'

        for symbol in stock_symbols:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
            response = requests.get(url)

            if response.status_code == 200:
                stock_data = response.json()
                file_name = f'raw_{symbol}_data.json'
                upload_to_gcs(bucket_name, file_name, json.dumps(stock_data))
                logging.info(f"Fetched and uploaded data for {symbol}")
            else:
                logging.error(f"Failed to fetch data for {symbol}. Status code: {response.status_code}")
        logging.info("Data extraction completed successfully")

    except Exception as e:
        logging.error(f"Error during extraction: {str(e)}")
        raise

# Parse task logic
def parse_data():
    logging.info("Starting data parsing")
    try:
        stock_symbols = ['AAPL', 'NFLX', 'MSFT', 'NVDA', 'AMZN']
        bucket_name = 'finnhub-financial-data'

        for symbol in stock_symbols:
            raw_file_name = f'raw_{symbol}_data.json'
            raw_data = download_from_gcs(bucket_name, raw_file_name)

            parsed_data = parse_stock_data(raw_data, symbol)
            if parsed_data:
                parsed_file_name = f'parsed_{symbol}_data.json'
                upload_to_gcs(bucket_name, parsed_file_name, json.dumps(parsed_data))
                logging.info(f"Parsed data uploaded for {symbol}")
            else:
                logging.error(f"No data parsed for {symbol}")

        logging.info("All data parsing and uploads completed successfully")

    except Exception as e:
        logging.error(f"Error during parsing: {str(e)}")
        raise

# Load task logic
def load_data():
    logging.info("Starting loading data into BigQuery")
    try:
        stock_symbols = ['AAPL', 'NFLX', 'MSFT', 'NVDA', 'AMZN']
        bucket_name = 'finnhub-financial-data'

        for symbol in stock_symbols:
            parsed_file_name = f'parsed_{symbol}_data.json'
            parsed_data = download_from_gcs(bucket_name, parsed_file_name)

            df = pd.DataFrame(parsed_data)
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            df['symbol'] = symbol

            df = df.drop_duplicates(subset=['date'])
            logging.info(f"{len(df)} rows remaining after dropping duplicates for {symbol}")

            table_id = f'finnhub-pipeline-ba882.financial_data.{symbol.lower()}_prices'
            client = bigquery.Client()
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND",
                autodetect=True,
            )

            job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
            job.result()
            logging.info(f"Loaded {len(df)} rows into {table_id}")

    except Exception as e:
        logging.error(f"Error during data load: {str(e)}")
        raise

# Default DAG arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
}

# Define DAG
dag = DAG(
    'finance_data_pipeline',
    default_args=default_args,
    description='Orchestrate extraction, parsing, and loading of stock data',
    schedule_interval='@daily',
)

# Define tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

parse_task = PythonOperator(
    task_id='parse_data',
    python_callable=parse_data,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

# Set task dependencies
extract_task >> parse_task >> load_task
