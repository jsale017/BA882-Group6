from flask import Flask, jsonify
import logging
from google.cloud import storage, bigquery
import json

app = Flask(__name__)

# Initialize logger
logging.basicConfig(level=logging.INFO)

# Function to download transformed data from Google Cloud Storage
def download_from_gcs(bucket_name, file_name):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        data = blob.download_as_string()
        logging.info(f"Downloaded data from {bucket_name}/{file_name}")
        return json.loads(data)
    except Exception as e:
        logging.error(f"Failed to download {file_name} from {bucket_name}: {e}")
        raise

# Function to check and create BigQuery table if it doesn't exist
def ensure_bigquery_table(dataset_id, table_id):
    try:
        client = bigquery.Client()
        dataset_ref = client.dataset(dataset_id)
        table_ref = dataset_ref.table(table_id)

        # Check if the table exists
        try:
            client.get_table(table_ref)
            logging.info(f"Table {table_id} already exists.")
        except Exception:
            # Define table schema
            schema = [
                bigquery.SchemaField("headline", "STRING"),
                bigquery.SchemaField("summary", "STRING"),
                bigquery.SchemaField("original_sentiment_score", "FLOAT"),
                bigquery.SchemaField("original_sentiment_label", "STRING"),
                bigquery.SchemaField("custom_sentiment_score", "FLOAT"),
                bigquery.SchemaField("custom_sentiment_label", "STRING"),
                bigquery.SchemaField("source", "STRING"),
                bigquery.SchemaField("published_at", "TIMESTAMP"),
            ]

            # Create the table
            table = bigquery.Table(table_ref, schema=schema)
            client.create_table(table)
            logging.info(f"Created table {table_id}.")
    except Exception as e:
        logging.error(f"Error ensuring BigQuery table: {e}")
        raise

# Function to load data into BigQuery
def load_to_bigquery(dataset_id, table_id, data):
    try:
        client = bigquery.Client()

        # Define the BigQuery table
        table_ref = client.dataset(dataset_id).table(table_id)

        # Format the data to match the BigQuery schema
        rows_to_insert = [
            {
                "headline": article["headline"],
                "summary": article["summary"],
                "original_sentiment_score": article.get("original_sentiment_score", None),
                "original_sentiment_label": article.get("original_sentiment_label", None),
                "custom_sentiment_score": article.get("custom_sentiment_score", None),
                "custom_sentiment_label": article.get("custom_sentiment_label", None),
                "source": article["source"],
                "published_at": article["published_at"][:4] + "-" + article["published_at"][4:6] + "-" +
                                article["published_at"][6:8] + "T" +
                                article["published_at"][9:11] + ":" + article["published_at"][11:13] + ":" +
                                article["published_at"][13:]
            }
            for article in data
        ]

        # Insert rows into BigQuery
        errors = client.insert_rows_json(table_ref, rows_to_insert)
        if errors:
            logging.error(f"Failed to insert rows into {table_id}: {errors}")
            raise Exception(f"BigQuery insert errors: {errors}")
        else:
            logging.info(f"Successfully loaded data into {table_id}")
    except Exception as e:
        logging.error(f"Error loading data into BigQuery: {e}")
        raise

# Main route to load news data into BigQuery
@app.route("/", methods=["GET", "POST"])
def load_news():
    logging.info("Starting loading of news data to BigQuery")
    try:
        # GCS bucket and filenames
        bucket_name = "finnhub-financial-data"
        stock_symbols = ['AAPL', 'NFLX', 'MSFT', 'NVDA', 'AMZN']

        # BigQuery configuration
        dataset_id = "financial_data"
        table_id = "news_sentiment"

        # Ensure the BigQuery table exists
        ensure_bigquery_table(dataset_id, table_id)

        for symbol in stock_symbols:
            # Download transformed data
            transformed_file_name = f"transformed_news_{symbol}_data.json"
            transformed_data = download_from_gcs(bucket_name, transformed_file_name)

            # Load transformed data into BigQuery
            load_to_bigquery(dataset_id, table_id, transformed_data)

        logging.info("News data successfully loaded to BigQuery")
        return jsonify({"message": "News data successfully loaded to BigQuery."}), 200

    except Exception as e:
        logging.error(f"Error during loading of news data: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

# Only include this if running locally (not necessary for deployment on Cloud Run)
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))  # Default to port 8080
    app.run(host="0.0.0.0", port=port)
