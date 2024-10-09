import logging
from google.cloud import secretmanager
from google.cloud import storage
import requests

# Initialize the logger
logging.basicConfig(level=logging.INFO)

# Function to get Alpha Vantage API key from Secret Manager
def get_alphavantage_api_key():
    client = secretmanager.SecretManagerServiceClient()
    name = "projects/finnhub-pipeline-ba882/secrets/alphavantage-api-key/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Function to upload data to GCS
def upload_to_gcs(bucket_name, data):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob('financial_data.json')
    blob.upload_from_string(data)
    logging.info(f"Uploaded data to {bucket_name}/financial_data.json")

# Main function to extract data and handle the request
def extract_data(request):
    logging.info("Starting data extraction")

    try:
        # Get API key
        alphavantage_api_key = get_alphavantage_api_key()
        logging.info("Successfully retrieved API key")

        # Fetch stock data from Alpha Vantage API
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey={alphavantage_api_key}"
        response = requests.get(url)
        
        if response.status_code == 200:
            logging.info("Data fetched successfully from Alpha Vantage")
            stock_data = response.json()

            # Upload stock data to GCS
            upload_to_gcs('finnhub-financial-data', str(stock_data))
            logging.info("Data upload complete")
            return "Data extraction and upload complete.", 200
        else:
            logging.error(f"Failed to fetch data from Alpha Vantage. Status code: {response.status_code}")
            return f"Error: Failed to fetch data. Status code: {response.status_code}", 500

    except Exception as e:
        logging.error(f"Error during data extraction: {str(e)}")
        return f"Error: {str(e)}", 500
