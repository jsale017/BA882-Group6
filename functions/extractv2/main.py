from flask import Flask, request, jsonify
import logging
from google.cloud import storage, secretmanager
import requests
import json
from datetime import datetime, timedelta
import time

app = Flask(__name__)

# Initializing logger
logging.basicConfig(level=logging.INFO)

# Function to get the Alpha Vantage API key from Google Secret Manager
def get_alphavantage_api_key():
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = "projects/finnhub-pipeline-ba882/secrets/alphavantage-api-key/versions/latest"
        response = client.access_secret_version(request={"name": name})
        logging.info("Successfully retrieved API key from Secret Manager")
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logging.error(f"Failed to retrieve API key: {e}")
        raise

# Function to upload raw data to Google Cloud Storage
def upload_to_gcs(bucket_name, file_name, data):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_string(data)
        logging.info(f"Uploaded raw data for {file_name} to {bucket_name}")
    except Exception as e:
        logging.error(f"Failed to upload {file_name} to {bucket_name}: {e}")
        raise

# Function to fetch stock data with retry logic
def fetch_stock_data(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                logging.info(f"Successfully fetched data from {url}")
                return response.json()
            else:
                logging.warning(f"Attempt {attempt + 1} failed: Status code {response.status_code} - {response.text}")
        except requests.RequestException as e:
            logging.error(f"Request error on attempt {attempt + 1} for URL {url}: {e}")
        time.sleep(2 ** attempt)  # Exponential backoff
    logging.error(f"Failed to fetch data from {url} after {max_retries} attempts")
    return None

# Main route to extract and upload raw data
@app.route("/", methods=["GET", "POST"])
def extract_data():
    logging.info("Starting data extraction")
    try:
        # Get the API key
        alphavantage_api_key = get_alphavantage_api_key()

        # List of stock symbols to fetch data for
        stock_symbols = ['AAPL', 'NFLX', 'MSFT', 'NVDA', 'AMZN']
        
        # Define your start date
        start_date = datetime(2023, 5, 1)

        # Fetching stock data from Alpha Vantage API
        for symbol in stock_symbols:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={alphavantage_api_key}&outputsize=full"
            stock_data = fetch_stock_data(url)
            
            if stock_data and "Meta Data" in stock_data:
                logging.info(f"Data fetched successfully for {symbol}")
                
                # Check the latest date in the response to confirm if it includes today’s data
                latest_date = max(stock_data["Time Series (Daily)"].keys())
                logging.info(f"Latest available date for {symbol} in API response: {latest_date}")

                # Filter data by start date
                filtered_data = {
                    "Meta Data": stock_data["Meta Data"],
                    "Time Series (Daily)": {
                        date: data for date, data in stock_data["Time Series (Daily)"].items()
                        if datetime.strptime(date, "%Y-%m-%d") >= start_date
                    }
                }

                # Log dates in the filtered data to confirm today’s data is included
                available_dates = list(filtered_data["Time Series (Daily)"].keys())
                logging.info(f"Available dates for {symbol} after filtering: {available_dates[:5]}... to {available_dates[-5:]}")
                
                # Check if today’s date is in the filtered data
                today_str = datetime.now().strftime("%Y-%m-%d")
                if today_str in filtered_data["Time Series (Daily)"]:
                    logging.info(f"Today's data for {symbol} is available and included.")
                else:
                    logging.warning(f"Today's data for {symbol} is not available in the API response.")

                # Upload filtered data to GCS
                upload_to_gcs('finnhub-financial-data', f'filtered_{symbol}_data.json', json.dumps(filtered_data))
            else:
                logging.error(f"Failed to fetch or missing 'Meta Data' for {symbol}. Response: {stock_data}")

        logging.info("Data extraction and upload complete")
        return jsonify({"message": "Data extraction and upload complete."}), 200

    except Exception as e:
        logging.error(f"Error during data extraction: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

# Only include this if running locally (not necessary for deployment on Cloud Run)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
