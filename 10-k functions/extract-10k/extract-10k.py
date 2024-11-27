from flask import Flask, jsonify
import logging
from google.cloud import storage
import requests
import json

app = Flask(__name__)

# Initialize logger
logging.basicConfig(level=logging.INFO)

# SEC EDGAR API Base URL
SEC_API_BASE = "https://data.sec.gov/submissions/CIK"

# Company CIKs
COMPANY_CIKS = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "NFLX": "0001065280",
    "AMZN": "0001018724"
}

# Function to fetch 10-K filings
def fetch_10k_filings(cik):
    url = f"{SEC_API_BASE}{int(cik):010d}.json"
    headers = {"User-Agent": "jsale017@bu.edu"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Failed to fetch data for CIK {cik}: {response.status_code}")
        return None

# Upload raw data to GCS
def upload_to_gcs(bucket_name, file_name, data):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(data, content_type="application/json")
    logging.info(f"Uploaded {file_name} to {bucket_name}")

@app.route("/", methods=["GET", "POST"])
def extract_data():
    try:
        bucket_name = "finnhub-financial-data"
        for symbol, cik in COMPANY_CIKS.items():
            logging.info(f"Fetching data for {symbol} (CIK: {cik})")
            filing_data = fetch_10k_filings(cik)
            if filing_data:
                file_name = f"raw_{symbol}_10k.json"
                upload_to_gcs(bucket_name, file_name, json.dumps(filing_data))
                logging.info(f"Raw data for {symbol} uploaded to GCS.")
            else:
                logging.warning(f"No data fetched for {symbol}.")
        return jsonify({"message": "Extraction completed."}), 200
    except Exception as e:
        logging.error(f"Extraction failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
