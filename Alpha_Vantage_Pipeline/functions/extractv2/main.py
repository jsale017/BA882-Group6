import logging
from google.cloud import storage, secretmanager
import requests
import json

# Initializing logger
logging.basicConfig(level=logging.INFO)

# Getting Alpha Vantage API key from Secret Manager
def get_alphavantage_api_key():
    client = secretmanager.SecretManagerServiceClient()
    name = "projects/finnhub-pipeline-ba882/secrets/alphavantage-api-key/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Uploading raw data to Google Cloud
def upload_to_gcs(bucket_name, file_name, data):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(data)
    logging.info(f"Uploaded raw data for {file_name} to {bucket_name}")

# Main function to extract and upload raw data
def extract_data(request):
    logging.info("Starting data extraction")

    try:
        # Getting API key
        alphavantage_api_key = get_alphavantage_api_key()
        logging.info("Successfully retrieved API key")

        stock_symbols = ['AAPL', 'NFLX', 'MSFT', 'NVDA', 'AMZN']

        # Fetching stock data from Alpha Vantage API
        for symbol in stock_symbols:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={alphavantage_api_key}"
            response = requests.get(url)

            if response.status_code == 200:
                logging.info(f"Data fetched successfully for {symbol}")
                stock_data = response.json()

                # Upload raw stock data to GCS with stock symbol in the filename
                upload_to_gcs('finnhub-financial-data', f'raw_{symbol}_data.json', json.dumps(stock_data))
            else:
                logging.error(f"Failed to fetch data for {symbol}. Status code: {response.status_code}")

        logging.info("All data extraction and uploads complete")
        return "Data extraction and upload complete.", 200

    except Exception as e:
        logging.error(f"Error during data extraction: {str(e)}")
        return f"Error: {str(e)}", 500
