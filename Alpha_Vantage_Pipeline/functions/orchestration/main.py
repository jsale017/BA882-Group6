from prefect import flow, task
import logging
import requests
import json
from google.cloud import storage, secretmanager, bigquery
import pandas as pd

# Task to get API key from Google Secret Manager
@task
def get_alphavantage_api_key():
    client = secretmanager.SecretManagerServiceClient()
    secret_name = "projects/finnhub-pipeline-ba882/secrets/alphavantage-api-key/versions/latest"
    response = client.access_secret_version(request={"name": secret_name})
    return response.payload.data.decode("UTF-8")

# Task to upload raw data to Google Cloud Storage
@task
def upload_to_gcs(bucket_name, file_name, data):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(data)
    logging.info(f"Uploaded {file_name} to bucket {bucket_name}")

# Task to download data from Google Cloud Storage
@task
def download_from_gcs(bucket_name, file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    raw_data = blob.download_as_string()
    logging.info(f"Downloaded {file_name} from bucket {bucket_name}")
    return json.loads(raw_data)

# Task to parse stock data
@task
def parse_stock_data(raw_data, symbol):
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
        logging.info(f"Successfully parsed stock data for {symbol}")
        return parsed_data
    except KeyError as e:
        logging.error(f"KeyError during parsing for {symbol}: {str(e)}")
        return None

# Task to extract data from Alpha Vantage API
@task
def extract_data(api_key):
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

# Task to parse and upload parsed data to GCS
@task
def parse_data():
    stock_symbols = ['AAPL', 'NFLX', 'MSFT', 'NVDA', 'AMZN']
    bucket_name = 'finnhub-financial-data'
    all_parsed_data = []

    for symbol in stock_symbols:
        raw_file_name = f'raw_{symbol}_data.json'
        raw_data = download_from_gcs(bucket_name, raw_file_name)
        parsed_data = parse_stock_data(raw_data, symbol)

        if parsed_data:
            parsed_file_name = f'parsed_{symbol}_data.json'
            upload_to_gcs(bucket_name, parsed_file_name, json.dumps(parsed_data))
            logging.info(f"Parsed data uploaded for {symbol}")
            all_parsed_data.extend(parsed_data)
        else:
            logging.error(f"No data parsed for {symbol}")

    logging.info(f"Total parsed data entries: {len(all_parsed_data)}")
    return all_parsed_data 

# Task to load data into BigQuery
@task
def load_data_to_bigquery(symbol, parsed_data):
    logging.info(f"Loading parsed data for {symbol} into BigQuery")

    df = pd.DataFrame(parsed_data)
    df['symbol'] = symbol 

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    table_id = f'finnhub-pipeline-ba882.financial_data.{symbol.lower()}_prices'
    client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        autodetect=True,
    )

    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()

    logging.info(f"Loaded data for {symbol} into {table_id}")

# Task to load combined data into a BigQuery table
@task
def load_combined_data_to_bigquery(all_parsed_data):
    if all_parsed_data:
        logging.info("Loading combined data into BigQuery")
        df = pd.DataFrame(all_parsed_data)

        numeric_columns = ["open", "high", "low", "close", "volume"]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

        combined_table_id = 'finnhub-pipeline-ba882.financial_data.all_stocks_prices'
        client = bigquery.Client()
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            autodetect=True,
        )

        job = client.load_table_from_dataframe(df, combined_table_id, job_config=job_config)
        job.result()  

        logging.info(f"Loaded combined data into {combined_table_id}")
    else:
        logging.error("No data to load into combined table")

# Main pipeline flow
@flow(name="main-pipeline")
def main_pipeline():
    # Step 1: Retrieve the API Key
    api_key = get_alphavantage_api_key()

    # Step 2: Extract data and upload to GCS
    extract_data(api_key)

    # Step 3: Parse raw data and upload parsed data to GCS
    all_parsed_data = parse_data()

    # Step 4: Load parsed data into BigQuery for each symbol
    for symbol in ['AAPL', 'NFLX', 'MSFT', 'NVDA', 'AMZN']:
        load_data_to_bigquery(symbol, all_parsed_data)

    # Step 5: Load combined data into BigQuery
    load_combined_data_to_bigquery(all_parsed_data)

# Run the main pipeline flow
if __name__ == "__main__":
    main_pipeline()
