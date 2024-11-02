from flask import Flask, request, jsonify
import logging
import json
from google.cloud import bigquery, storage
import pandas as pd

app = Flask(__name__)

# Initialize logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    logger.info("Health check endpoint accessed.")
    return jsonify({"status": "healthy"}), 200

# Function to download data from Google Cloud Storage
def download_from_gcs(bucket_name, file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    raw_data = blob.download_as_string()
    logger.info(f"Downloaded data from {bucket_name}/{file_name}")
    return json.loads(raw_data)

# Function to load data into BigQuery
def load_data_to_bigquery(data, staging_table_id, final_table_id, all_stocks_table_id="finnhub-pipeline-ba882.financial_data.all_stocks_prices"):
    logger.info("Starting BigQuery load process.")
    try:
        # Convert data to DataFrame and clean
        df = pd.DataFrame(data)
        logger.info("Data converted to DataFrame.")
        
        # Convert numeric columns to appropriate types
        df['volume'] = df['volume'].astype(pd.Int64Dtype(), errors='ignore')
        numeric_columns = ['open', 'high', 'low', 'close']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Drop duplicates and clean data
        df = df.dropna(subset=['symbol']).drop_duplicates(subset=['date', 'open', 'close', 'high', 'low', 'volume', 'symbol'])
        
        logger.info("Data cleaned and prepared for BigQuery.")
        client = bigquery.Client()
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)
        
        # Load data into staging table
        logger.info(f"Loading data into staging table: {staging_table_id}")
        job = client.load_table_from_dataframe(df, staging_table_id, job_config=job_config)
        job.result(timeout=300)
        logger.info(f"Successfully loaded data into {staging_table_id}")
        
        # Merge data into final table
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
            T.volume = S.volume,
            T.symbol = S.symbol
        WHEN NOT MATCHED THEN
          INSERT (date, open, high, low, close, volume, symbol)
          VALUES (S.date, S.open, S.high, S.low, S.close, S.volume, S.symbol)
        """
        logger.info(f"Merging data from {staging_table_id} into {final_table_id}")
        query_job = client.query(merge_query)
        query_job.result()
        logger.info(f"Successfully merged data into {final_table_id}")
        
        # Merge into all_stocks_prices table
        all_stocks_merge_query = f"""
        MERGE `{all_stocks_table_id}` T
        USING `{staging_table_id}` S
        ON T.date = S.date AND T.symbol = S.symbol
        WHEN MATCHED THEN
          UPDATE SET
            T.open = S.open,
            T.high = S.high,
            T.low = S.low,
            T.close = S.close,
            T.volume = S.volume
        WHEN NOT MATCHED THEN
          INSERT (date, open, high, low, close, volume, symbol)
          VALUES (S.date, S.open, S.high, S.low, S.close, S.volume, S.symbol)
        """
        logger.info(f"Merging data from {staging_table_id} into {all_stocks_table_id}")
        all_stocks_query_job = client.query(all_stocks_merge_query)
        all_stocks_query_job.result()
        logger.info(f"Successfully merged data into {all_stocks_table_id}")

    except Exception as e:
        logger.error(f"BigQuery loading error: {str(e)}")
        raise

# Main route for the load process
@app.route("/", methods=["GET", "POST"])
def load_data():
    logger.info("Starting data load process for multiple stocks.")
    try:
        stock_symbols = ['AAPL', 'NFLX', 'MSFT', 'NVDA', 'AMZN']
        bucket_name = 'finnhub-financial-data'

        for symbol in stock_symbols:
            file_name = f'parsed_{symbol}_data.json'
            data = download_from_gcs(bucket_name, file_name)
            staging_table_id = f'finnhub-pipeline-ba882.financial_data.{symbol.lower()}_prices_staging'
            final_table_id = f'finnhub-pipeline-ba882.financial_data.{symbol.lower()}_prices'
            load_data_to_bigquery(data, staging_table_id, final_table_id)

        logger.info("All data successfully loaded into BigQuery.")
        return jsonify({"message": "Data successfully loaded into BigQuery."}), 200

    except Exception as e:
        logger.error(f"Error during data load: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting Flask application on port 8080")
    app.run(host="0.0.0.0", port=8080)
