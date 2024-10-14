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

# Loading data into a staging table and merging into the final table in BigQuery
def load_data_to_bigquery(data, staging_table_id, final_table_id):
    # Converting parsed data to DataFrame
    df = pd.DataFrame(data)
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Initializing BigQuery client
    client = bigquery.Client()

    # Load configuration for the staging table
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",  # Replace the staging table
        autodetect=True,  # Automatically infer schema
    )

    # Load data into the staging table
    logging.info(f"Loading data into staging table: {staging_table_id}")
    job = client.load_table_from_dataframe(df, staging_table_id, job_config=job_config)
    job.result()  # Wait for the load job to complete

    logging.info(f"Successfully loaded data into {staging_table_id}")

    # MERGE query to merge data from the staging table to the final table
    merge_query = f"""
    MERGE `{final_table_id}` T
    USING `{staging_table_id}` S
    ON T.date = S.date
    WHEN MATCHED THEN
      UPDATE SET
        T.open = S.open,
        T.high = S.high,
        T.low = S.low,
        T.close = S.close,
        T.volume = S.volume
    WHEN NOT MATCHED THEN
      INSERT (date, open, high, low, close, volume)
      VALUES (S.date, S.open, S.high, S.low, S.close, S.volume)
    """

    # Execute the MERGE query
    logging.info(f"Merging data from {staging_table_id} into {final_table_id}")
    query_job = client.query(merge_query)
    query_job.result()  # Wait for the query to complete

    logging.info(f"Successfully merged data into {final_table_id}")

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

            try:
                # Download the parsed data from GCS
                data = download_from_gcs(bucket_name, file_name)
                logging.info(f"Successfully downloaded parsed data for {symbol}.")

                # Define the staging and final table IDs
                staging_table_id = f'finnhub-pipeline-ba882.financial_data.{symbol.lower()}_prices_staging'
                final_table_id = f'finnhub-pipeline-ba882.financial_data.{symbol.lower()}_prices'

                # Load data to the staging table and merge it into the final table
                load_data_to_bigquery(data, staging_table_id, final_table_id)

            except Exception as e:
                logging.error(f"Error processing {symbol}: {str(e)}")
                continue

        logging.info("All data successfully loaded into BigQuery")
        return "Data successfully loaded into BigQuery.", 200

    except Exception as e:
        logging.error(f"Error during data load: {str(e)}")
        return f"Error: {str(e)}", 500
