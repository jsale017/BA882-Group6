import functions_framework
from google.cloud import secretmanager, bigquery, storage
import pandas as pd
import logging

# Directly specify project and storage details
project_id = 'finnhub-pipeline-ba882'
bucket_name = 'alpha_vatange_vertex_models'
ml_dataset_path = 'training-data/stocks/'

# Initialize BigQuery and Storage clients
bigquery_client = bigquery.Client(project=project_id)
storage_client = storage.Client(project=project_id)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Function to retrieve the API key from Google Secret Manager
def get_secret_key():
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/alphavantage-api-key/versions/latest"
        response = client.access_secret_version(request={"name": name})
        logging.info("Successfully retrieved API key from Secret Manager")
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logging.error(f"Error retrieving API key from Secret Manager: {e}")
        raise

# Function to upload cleaned data to Google Cloud Storage
def upload_to_gcs(bucket_name, destination_path, data):
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_path)
        blob.upload_from_string(data.to_csv(index=False), content_type="text/csv")
        logging.info(f"Cleaned data successfully uploaded to GCS at {destination_path}")
    except Exception as e:
        logging.error(f"Failed to upload cleaned data to GCS: {e}")
        raise

@functions_framework.http
def task(request):
    try:
        # Retrieve the API key from Secret Manager
        alphavantage_api_key = get_secret_key()

        # Query to fetch stock data from BigQuery
        query = """
            SELECT *
            FROM `finnhub-pipeline-ba882.financial_data.trades`
        """
        logging.info("Fetching data from BigQuery")
        df = bigquery_client.query(query).to_dataframe()

        # Clean the data
        logging.info("Cleaning the data")
        df.dropna(thresh=len(df) * 0.2, axis=1, inplace=True)
        df.dropna(inplace=True)

        # Define the path for saving the cleaned data
        cleaned_data_path = f"{ml_dataset_path}cleaned_stock_data.csv"

        # Upload cleaned data to Google Cloud Storage
        upload_to_gcs(bucket_name, cleaned_data_path, df)

        # Respond with the path to the cleaned data
        return {"cleaned_data_path": f"gs://{bucket_name}/{cleaned_data_path}"}, 200

    except Exception as e:
        logging.error(f"An error occurred during the task: {e}")
        return {"error": str(e)}, 500
