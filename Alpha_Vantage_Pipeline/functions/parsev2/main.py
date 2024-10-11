import logging
import json
from google.cloud import storage
import pandas as pd
import functions_framework

# Initializing logger
logging.basicConfig(level=logging.INFO)

# Downloading raw data from Google Cloud
def download_from_gcs(bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob('raw_financial_data.json')  # Grabbing raw_financial_data.json
    raw_data = blob.download_as_string()
    logging.info(f"Downloaded raw data from {bucket_name}/raw_financial_data.json")
    return json.loads(raw_data)

# Uploading parsed data to GCS
def upload_parsed_data_to_gcs(bucket_name, data):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob('parsed_financial_data.json')  # Uploading parsed data as parsed_financial_data.json
    blob.upload_from_string(data)
    logging.info(f"Uploaded parsed data to {bucket_name}/parsed_financial_data.json")

# Parsing raw Alpha Vantage data
def parse_stock_data(raw_data):
    try:
        # Extracting  Daily Time Series
        time_series = raw_data.get("Time Series (Daily)", {})

        # Create a list of parsed records
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
        # Downloading raw data
        raw_data = download_from_gcs('finnhub-financial-data')

        # Parsing the raw data
        parsed_data = parse_stock_data(raw_data)
        if parsed_data is not None:
            # Converting parsed data to JSON
            parsed_data_json = json.dumps(parsed_data)

            # Uploading parsed data to GCS
            upload_parsed_data_to_gcs('finnhub-financial-data', parsed_data_json)
            logging.info("Parsed data upload complete")
            return "Data parsing and upload complete.", 200
        else:
            logging.error("Parsing failed")
            return "Error: Parsing failed.", 500

    except Exception as e:
        logging.error(f"Error during data parsing: {str(e)}")
        return f"Error: {str(e)}", 500
