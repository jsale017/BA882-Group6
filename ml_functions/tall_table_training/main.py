import functions_framework
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
import logging
import io  # Importing io for StringIO

# Set up logging
logging.basicConfig(level=logging.INFO)

# Constants
BUCKET_NAME = 'alpha_vatange_vertex_models'  # Replace with your bucket name
SOURCE_FILE_PATH = 'training-data/stocks/cleaned_stock_data.csv'  # Path to the cleaned data file
DESTINATION_FOLDER = 'training-data/stocks/split_data/'  # Destination folder in GCS for split data files

# Initialize Cloud Storage client
storage_client = storage.Client()

# Function to download and load CSV data from GCS
def load_data_from_gcs(bucket_name, file_path):
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        data = blob.download_as_text()
        # Using io.StringIO to convert text data into a format readable by pd.read_csv
        df = pd.read_csv(io.StringIO(data))
        logging.info("Data loaded successfully from GCS.")
        return df
    except Exception as e:
        logging.error(f"Error loading data from GCS: {e}")
        raise

# Function to split the data into training, test, and validation sets
def split_data(df):
    try:
        train, temp = train_test_split(df, test_size=0.2, random_state=42)
        test, validation = train_test_split(temp, test_size=0.5, random_state=42)
        logging.info("Data split into train, test, and validation sets.")
        return train, test, validation
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

# Function to upload DataFrame as CSV to GCS
def upload_to_gcs(df, bucket_name, destination_path):
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_path)
        blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")
        logging.info(f"Uploaded {destination_path} to GCS.")
    except Exception as e:
        logging.error(f"Error uploading {destination_path} to GCS: {e}")
        raise

# Main pipeline function
@functions_framework.http
def stock_data_pipeline(request):
    try:
        # Load data from GCS
        data = load_data_from_gcs(BUCKET_NAME, SOURCE_FILE_PATH)
        
        # Split data
        train_data, test_data, validation_data = split_data(data)
        
        # Define file paths for each split
        train_path = f"{DESTINATION_FOLDER}train_stock_data.csv"
        test_path = f"{DESTINATION_FOLDER}test_stock_data.csv"
        validation_path = f"{DESTINATION_FOLDER}validation_stock_data.csv"
        
        # Upload each split dataset to GCS
        upload_to_gcs(train_data, BUCKET_NAME, train_path)
        upload_to_gcs(test_data, BUCKET_NAME, test_path)
        upload_to_gcs(validation_data, BUCKET_NAME, validation_path)
        
        # Return a success response
        return {"message": "Data split and uploaded successfully."}, 200

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        return {"error": str(e)}, 500
