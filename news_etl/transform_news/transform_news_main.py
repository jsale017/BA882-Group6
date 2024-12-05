from flask import Flask, jsonify
import logging
from google.cloud import storage
from textblob import TextBlob
import json

app = Flask(__name__)

# Initialize logger
logging.basicConfig(level=logging.INFO)

# Function to download raw data from Google Cloud Storage
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

# Function to upload transformed data to Google Cloud Storage
def upload_to_gcs(bucket_name, file_name, data):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_string(json.dumps(data))
        logging.info(f"Uploaded transformed data to {bucket_name}/{file_name}")
    except Exception as e:
        logging.error(f"Failed to upload {file_name} to {bucket_name}: {e}")
        raise

# Function to perform sentiment analysis
def analyze_sentiment(news_data):
    transformed_data = []
    for article in news_data:
        headline = article.get("title", "")
        summary = article.get("summary", "")
        sentiment_score = TextBlob(summary).sentiment.polarity
        sentiment_label = (
            "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
        )
        transformed_data.append({
            "headline": headline,
            "summary": summary,
            "original_sentiment_score": article.get("overall_sentiment_score", 0),
            "original_sentiment_label": article.get("overall_sentiment_label", "Unknown"),
            "custom_sentiment_score": sentiment_score,
            "custom_sentiment_label": sentiment_label,
            "source": article.get("source", ""),
            "published_at": article.get("time_published", ""),
        })
    return transformed_data

# Main route to transform news data
@app.route("/", methods=["GET", "POST"])
def transform_news():
    logging.info("Starting news data transformation")
    try:
        # GCS bucket and filenames
        bucket_name = "finnhub-financial-data"
        stock_symbols = ['AAPL', 'NFLX', 'MSFT', 'NVDA', 'AMZN']

        for symbol in stock_symbols:
            # Download raw data
            raw_file_name = f"news_{symbol}_data.json"
            raw_data = download_from_gcs(bucket_name, raw_file_name)

            # Perform sentiment analysis
            transformed_data = analyze_sentiment(raw_data)

            # Upload transformed data
            transformed_file_name = f"transformed_news_{symbol}_data.json"
            upload_to_gcs(bucket_name, transformed_file_name, transformed_data)

        logging.info("News data transformation complete")
        return jsonify({"message": "News data transformation complete."}), 200

    except Exception as e:
        logging.error(f"Error during news data transformation: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

# Only include this if running locally (not necessary for deployment on Cloud Run)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
