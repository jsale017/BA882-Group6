from flask import Flask, jsonify
import logging
from google.cloud import storage, secretmanager
import requests
import json
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

# Function to fetch news sentiment data with retry logic
def fetch_news_data(url, max_retries=3):
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

# Main route to extract and upload news sentiment data
@app.route("/", methods=["GET", "POST"])
def extract_news():
    logging.info("Starting news sentiment extraction")
    try:
        # Get the API key
        alphavantage_api_key = get_alphavantage_api_key()

        # List of stock symbols to fetch news for
        stock_symbols = ['AAPL', 'NFLX', 'MSFT', 'NVDA', 'AMZN']
        
        # Fetching news sentiment data from Alpha Vantage API
        for symbol in stock_symbols:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={alphavantage_api_key}"
            news_data = fetch_news_data(url)
            
            if news_data and "feed" in news_data:
                logging.info(f"News data fetched successfully for {symbol}")

                # Extract the news feed
                news_feed = news_data["feed"]
                logging.info(f"Number of articles fetched for {symbol}: {len(news_feed)}")

                # Upload news feed to GCS
                upload_to_gcs('finnhub-financial-data', f'news_{symbol}_data.json', json.dumps(news_feed))
            else:
                logging.error(f"Failed to fetch or missing 'feed' for {symbol}. Response: {news_data}")

        logging.info("News sentiment extraction and upload complete")
        return jsonify({"message": "News sentiment extraction and upload complete."}), 200

    except Exception as e:
        logging.error(f"Error during news sentiment extraction: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

# Only include this if running locally (not necessary for deployment on Cloud Run)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
