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
    
    # Ensure the numeric columns are properly converted to float
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Converting columns to numeric, coercing errors to NaN (if any strings are found)
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
    logging.info("Starting data load process")

    try:
        bucket_name = 'finnhub-financial-data'
        file_name = 'parsed_financial_data.json'  # Grabbing parsed data from GCS
        
        # Setting up BigQuery Table ID
        table_id = 'finnhub-pipeline-ba882.financial_data.stock_prices'

        
        # Downloading the parsed data from GCS
        data = download_from_gcs(bucket_name, file_name)
        
        # Loading the data into BigQuery
        load_data_to_bigquery(data, table_id)
        
        return "Data successfully loaded into BigQuery.", 200

    except Exception as e:
        logging.error(f"Error during data load: {str(e)}")
        return f"Error: {str(e)}", 500
