from flask import Flask, jsonify
import logging
import os
from google.cloud import storage, bigquery
import json
from datetime import datetime

app = Flask(__name__)

# Initialize logger
logging.basicConfig(level=logging.INFO)

# GCS and BigQuery configuration
BUCKET_NAME = "finnhub-financial-data"
BQ_DATASET = "financial_data"
BQ_TABLE = "10k_filings"

# Company CIKs
COMPANY_CIKS = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "NFLX": "0001065280",
    "AMZN": "0001018724"
}

# Initialize BigQuery client
bigquery_client = bigquery.Client()

# Function to load JSON data from GCS
def load_from_gcs(file_name):
    """Load JSON data from GCS."""
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(file_name)
        data = blob.download_as_text()
        logging.info(f"Successfully loaded {file_name} from GCS.")
        return json.loads(data)
    except Exception as e:
        logging.error(f"Error loading {file_name} from GCS: {e}")
        return None

# Function to parse and transform JSON data for BigQuery
def transform_data(company_name, cik, raw_data):
    """Transform raw JSON data into BigQuery-compatible rows."""
    rows = []
    try:
        filings = raw_data.get("filings", {}).get("recent", {})
        for i in range(len(filings.get("form", []))):
            filing_date = filings.get("filingDate", [None])[i]
            row = {
                "company_name": company_name,
                "cik": cik,
                "form_type": filings["form"][i],
                "filing_date": filing_date if not filing_date else str(filing_date),
                "document_links": filings.get("primaryDocument", [])[i].split(),
                "description": filings.get("primaryDocDescription", [])[i],
                "content": None,  # Content extraction can be added if available
            }
            rows.append(row)
    except Exception as e:
        logging.error(f"Error transforming data for {company_name}: {e}")
    return rows

# Function to load data into BigQuery
def load_into_bigquery(rows):
    """Load rows into BigQuery."""
    try:
        table_id = f"{bigquery_client.project}.{BQ_DATASET}.{BQ_TABLE}"
        errors = bigquery_client.insert_rows_json(table_id, rows)
        if errors:
            logging.error(f"Errors occurred while loading data into BigQuery: {errors}")
        else:
            logging.info(f"Successfully loaded {len(rows)} rows into {table_id}.")
    except Exception as e:
        logging.error(f"Error loading data into BigQuery: {e}")

@app.route("/", methods=["GET", "POST"])
def load_data():
    """Load raw 10-K filings data from GCS into BigQuery."""
    try:
        for symbol, cik in COMPANY_CIKS.items():
            file_name = f"raw_{symbol}_10k.json"
            logging.info(f"Processing {file_name}")
            raw_data = load_from_gcs(file_name)
            if raw_data:
                rows = transform_data(symbol, cik, raw_data)
                load_into_bigquery(rows)
            else:
                logging.warning(f"No data found for {file_name}")
        return jsonify({"message": "Loading completed."}), 200
    except Exception as e:
        logging.error(f"Loading failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask app
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
