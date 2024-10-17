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
    blob = bucket.blob(file_name)
    blob.upload_from_string(data)
    logging.info(f"Uploaded parsed data to {bucket_name}/{file_name}")

# Function to clean parsed stock data
def clean_parsed_data(df, symbol):
    df['symbol'] = df['symbol'].fillna(symbol)

    df = df.drop_duplicates(subset=['date', 'open', 'close', 'high', 'low', 'volume', 'symbol'])

    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    df = df.dropna(subset=['open', 'close', 'symbol'])

    return df

# Parsing stock data
def parse_stock_data(raw_data, symbol):
    try:
        # Extracting daily time series
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

        # Converting parsed data to DataFrame
        df = pd.DataFrame(parsed_data)

        # Cleaning the parsed data
        df = clean_parsed_data(df, symbol)

        logging.info(f"Successfully parsed and cleaned stock data for {symbol}")
        return df

    except KeyError as e:
        logging.error(f"KeyError during parsing for {symbol}: {str(e)}")
        return None

# Main function to handle HTTP requests
@functions_framework.http
def parse_data(request):
    logging.info("Starting data parsing")

    try:
        # Listing stock symbols
        stock_symbols = ['AAPL', 'NFLX', 'MSFT', 'NVDA', 'AMZN']

        for symbol in stock_symbols:
            file_name = f'raw_{symbol}_data.json'
            raw_data = download_from_gcs('finnhub-financial-data', file_name)

            # Parsing the raw data
            parsed_data = parse_stock_data(raw_data, symbol)
            if parsed_data is not None:
                parsed_file_name = f'parsed_{symbol}_data.json'
                upload_parsed_data_to_gcs('finnhub-financial-data', parsed_file_name, parsed_data.to_json(orient="records"))
        
        logging.info("All data parsing and uploads complete")
        return "Data parsing and upload complete.", 200

    except Exception as e:
        logging.error(f"Error during data parsing: {str(e)}")
        return f"Error: {str(e)}", 500
