import logging
import json
from google.cloud import bigquery, storage
import pandas as pd
import functions_framework

# Initializing logger
logging.basicConfig(level=logging.INFO)

# Obtaining parsed data from GCS
def download_from_gcs(bucket_name, file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    raw_data = blob.download_as_string()
    logging.info(f"Downloaded data from {bucket_name}/{file_name}")
    return json.loads(raw_data)

# Loading data into BigQuery
def load_data_to_bigquery(data, table_id):
    # Converting parsed data to DataFrame
    df = pd.DataFrame(data)
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Initializing BigQuery client
    client = bigquery.Client()
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",  # Appending data to the table
        autodetect=True,  # Automatically infering the schema
    )
    
    # Loading data to BigQuery
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    logging.info(f"Loaded {len(df)} rows into {table_id}.")

# Main function to handle the HTTP request
@functions_framework.http
def load_data(request):
    logging.info("Starting data load process for multiple stocks")

    try:
        stock_symbols = ['AAPL', 'NFLX', 'MSFT', 'NVDA', 'AMZN']
        bucket_name = 'finnhub-financial-data'

        for symbol in stock_symbols:
            file_name = f'parsed_{symbol}_data.json'
            logging.info(f"Processing {symbol}: Checking file {file_name} in GCS...")

            # Check if the file exists in GCS
            try:
                data = download_from_gcs(bucket_name, file_name)
                logging.info(f"Successfully downloaded parsed data for {symbol}.")
                # Set BigQuery table for each stock
                table_id = f'finnhub-pipeline-ba882.financial_data.{symbol.lower()}_prices'
                # Load the parsed data into BigQuery
                load_data_to_bigquery(data, table_id)
                logging.info(f"Loaded data into {table_id} for {symbol}.")
            except Exception as e:
                logging.error(f"Error processing {symbol}: {str(e)}")
                continue

        logging.info("All data successfully loaded into BigQuery")
        return "Data successfully loaded into BigQuery.", 200

    except Exception as e:
        logging.error(f"Error during data load: {str(e)}")
        return f"Error: {str(e)}", 500
