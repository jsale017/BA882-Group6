import logging
from google.cloud import storage, secretmanager
import requests
import json

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
    blob = bucket.blob('parsed_financial_data.json')  # Changed file name to indicate parsed data
    blob.upload_from_string(data)
    logging.info(f"Uploaded parsed data to {bucket_name}/parsed_financial_data.json")

# Function to parse the raw Alpha Vantage response data
def parse_stock_data(raw_data):
    try:
        # Extract the 'Time Series (Daily)' section
        time_series = raw_data.get("Time Series (Daily)", {})

        # Create a list of parsed records
        parsed_data = []
        
        # Loop through each date and extract key information
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

# Main function to extract, parse, and upload data
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

            # Parse the raw stock data
            parsed_data = parse_stock_data(stock_data)
            if parsed_data is not None:
                # Convert the parsed data to a JSON string
                parsed_data_json = json.dumps(parsed_data)

                # Upload parsed data to GCS
                upload_to_gcs('finnhub-financial-data', parsed_data_json)
                logging.info("Data upload complete")
                return "Data extraction, parsing, and upload complete.", 200
            else:
                logging.error("Parsing failed")
                return "Error: Parsing failed.", 500
        else:
            logging.error(f"Failed to fetch data from Alpha Vantage. Status code: {response.status_code}")
            return f"Error: Failed to fetch data. Status code: {response.status_code}", 500

    except Exception as e:
        logging.error(f"Error during data extraction: {str(e)}")
        return f"Error: {str(e)}", 500
