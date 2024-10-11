import logging
import json
from google.cloud import storage
import pandas as pd
import functions_framework

# Initializing logger
logging.basicConfig(level=logging.INFO)

# Downloading raw data from Google Cloud
def download_from_gcs(bucket_name, file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)  # Grabbing raw data per stock
    raw_data = blob.download_as_string()
    logging.info(f"Downloaded raw data from {bucket_name}/{file_name}")
    return json.loads(raw_data)

# Uploading parsed data to GCS
def upload_parsed_data_to_gcs(bucket_name, file_name, data):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)  # Uploading parsed data
    blob.upload_from_string(data)
    logging.info(f"Uploaded parsed data to {bucket_name}/{file_name}")

# Parsing raw Alpha Vantage data
def parse_stock_data(raw_data):
    try:
        # Extracting  Daily Time Series
        time_series = raw_data.get("Time Series (Daily)", {})
        parsed_data = []
        
        # Looping through each date and extract key information
        for date, daily_data in time_series.items():
            parsed_record = {
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

# Main function to handle HTTP requests
@functions_framework.http
def parse_data(request):
    logging.info("Starting data parsing")

    try:
        stock_symbols = ['AAPL', 'NFLX', 'MSFT', 'NVDA', 'AMZN']

        for symbol in stock_symbols:
            file_name = f'raw_{symbol}_data.json'
            raw_data = download_from_gcs('finnhub-financial-data', file_name)

            # Parsing the raw data
            parsed_data = parse_stock_data(raw_data)
            if parse_data is not None:
                parsed_file_name = f'parsed_{symbol}_data.json'
                upload_parsed_data_to_gcs('finnhub-financial-data', parsed_file_name, json.dumps(parsed_data))
        
        logging.info("All data parsing and uploads complete")
        return "Data parsing and upload complete.", 200

    except Exception as e:
        logging.error(f"Error during data parsing: {str(e)}")
        return f"Error: {str(e)}", 500