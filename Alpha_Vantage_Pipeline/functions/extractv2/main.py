from flask import Flask, request, jsonify
import logging
from google.cloud import storage, secretmanager
import requests
import json
from datetime import datetime  # Moved here with other imports

app = Flask(__name__)

# Initializing logger
logging.basicConfig(level=logging.INFO)

# Function to get the Alpha Vantage API key from Google Secret Manager
def get_alphavantage_api_key():
    client = secretmanager.SecretManagerServiceClient()
    name = "projects/finnhub-pipeline-ba882/secrets/alphavantage-api-key/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Function to upload raw data to Google Cloud Storage
def upload_to_gcs(bucket_name, file_name, data):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(data)
    logging.info(f"Uploaded raw data for {file_name} to {bucket_name}")

# Main route to extract and upload raw data
@app.route("/", methods=["GET", "POST"])
def extract_data():
    logging.info("Starting data extraction")
    try:
        # Get the API key
        alphavantage_api_key = get_alphavantage_api_key()
        logging.info("Successfully retrieved API key")

        # List of stock symbols to fetch data for
        stock_symbols = ['AAPL', 'NFLX', 'MSFT', 'NVDA', 'AMZN']
        
        # Define your start date
        start_date = datetime(2023, 5, 1)

        # Fetching stock data from Alpha Vantage API
        for symbol in stock_symbols:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={alphavantage_api_key}&outputsize=full"
            response = requests.get(url)
            
            if response.status_code == 200:
                logging.info(f"Data fetched successfully for {symbol}")
                stock_data = response.json()
                
                # Filter data by start date
                filtered_data = {
                    "Meta Data": stock_data["Meta Data"],
                    "Time Series (Daily)": {
                        date: data for date, data in stock_data["Time Series (Daily)"].items()
                        if datetime.strptime(date, "%Y-%m-%d") >= start_date
                    }
                }

                # Upload filtered data to GCS
                upload_to_gcs('finnhub-financial-data', f'filtered_{symbol}_data.json', json.dumps(filtered_data))
            else:
                logging.error(f"Failed to fetch data for {symbol}. Status code: {response.status_code}")

        logging.info("Data extraction and upload complete")
        return jsonify({"message": "Data extraction and upload complete."}), 200

    except Exception as e:
        logging.error(f"Error during data extraction: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Only include this if running locally (not necessary for deployment on Cloud Run)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
