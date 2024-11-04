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
def load_data_to_bigquery(data):
    logger.info("Starting BigQuery load process.")
    try:
        # Set the target table ID directly
        trades_table_id = "finnhub-pipeline-ba882.financial_data.trades"
        
        # Convert data to DataFrame and clean
        df = pd.DataFrame(data)
        logger.info("Data converted to DataFrame.")
        
        # Rename 'date' to 'trade_date' to match BigQuery schema
        df.rename(columns={'date': 'trade_date'}, inplace=True)
        
        # Convert 'trade_date' column to datetime format
        df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce').dt.date

        # Convert numeric columns to appropriate types
        df['volume'] = df['volume'].fillna(0).astype('Int64')
        numeric_columns = ['open', 'high', 'low', 'close']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Drop duplicates and clean data
        df = df.dropna(subset=['symbol']).drop_duplicates(subset=['trade_date', 'open', 'close', 'high', 'low', 'volume', 'symbol'])
        
        logger.info("Data cleaned and prepared for BigQuery.")
        
        # Set up BigQuery client and job configuration
        client = bigquery.Client()
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND", autodetect=True)
        
        # Load data directly into the trades table
        logger.info(f"Loading data into trades table: {trades_table_id}")
        job = client.load_table_from_dataframe(df, trades_table_id, job_config=job_config)
        job.result(timeout=300)
        logger.info(f"Successfully loaded data into {trades_table_id}")

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
            load_data_to_bigquery(data)  # Call without trades_table_id

        logger.info("All data successfully loaded into BigQuery.")
        return jsonify({"message": "Data successfully loaded into BigQuery."}), 200

    except Exception as e:
        logger.error(f"Error during data load: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting Flask application on port 8080")
    app.run(host="0.0.0.0", port=8080)
