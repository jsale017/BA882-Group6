from flask import Flask, request, jsonify
import logging
import json
from google.cloud import storage
import pandas as pd

app = Flask(__name__)

# Initializing logger
logging.basicConfig(level=logging.INFO)

# Function to download raw data from Google Cloud Storage
def download_from_gcs(bucket_name, file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    raw_data = blob.download_as_string()
    logging.info(f"Downloaded raw data from {bucket_name}/{file_name}")
    return json.loads(raw_data)

# Function to upload parsed data to Google Cloud Storage
def upload_parsed_data_to_gcs(bucket_name, file_name, data):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    if blob.exists():
        logging.info(f"Deleting existing file {bucket_name}/{file_name}")
        blob.delete()
    blob.upload_from_string(data, content_type="application/json")
    logging.info(f"Uploaded parsed data to {bucket_name}/{file_name}")

# Function to parse stock data and convert to DataFrame
def parse_stock_data(raw_data, symbol):
    # Extracting the time series data and converting it into a list of dictionaries
    time_series = raw_data.get("Time Series (Daily)", {})
    parsed_data = []
    for date, daily_data in time_series.items():
        try:
            # Attempt to retrieve and convert volume to float
            volume = daily_data.get("6. volume")
            if volume is not None:
                volume = float(volume)
            else:
                volume = None  # Handle missing volume as None
            
            # Parsing each day's stock data
            parsed_data.append({
                "symbol": symbol,
                "date": date,
                "open": float(daily_data.get("1. open", "nan")),
                "high": float(daily_data.get("2. high", "nan")),
                "low": float(daily_data.get("3. low", "nan")),
                "close": float(daily_data.get("4. close", "nan")),
                "volume": volume
            })
        except ValueError as e:
            logging.warning(f"Data conversion error for {symbol} on {date}: {e}")
    
    # Converting the list of dictionaries to a DataFrame
    df = pd.DataFrame(parsed_data)
    logging.info(f"Parsed data for {symbol}:\n{df.head()}")
    return df

# Main endpoint to parse data
@app.route("/", methods=["GET", "POST"])
def parse_data():
    logging.info("Starting data parsing")
    try:
        # List of stock symbols
        stock_symbols = ['AAPL', 'NFLX', 'MSFT', 'NVDA', 'AMZN']
        for symbol in stock_symbols:
            logging.info(f"Processing symbol: {symbol}")
            file_name = f'filtered_{symbol}_data.json'
            
            # Downloading raw data
            raw_data = download_from_gcs('finnhub-financial-data', file_name)
            parsed_df = parse_stock_data(raw_data, symbol)

            # Uploading parsed data as JSON
            parsed_file_name = f'parsed_{symbol}_data.json'
            upload_parsed_data_to_gcs(
                'finnhub-financial-data',
                parsed_file_name,
                parsed_df.to_json(orient="records")
            )
            logging.info(f"Uploaded parsed data for {symbol} to {parsed_file_name}")

        logging.info("Data parsing and upload complete")
        return jsonify({"message": "Data parsing and upload complete."}), 200

    except Exception as e:
        logging.error(f"Error during data parsing: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
