import functions_framework
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
import logging
import io
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)

# Constants
BUCKET_NAME = 'alpha_vatange_vertex_models'
SOURCE_FILE_PATH = 'training-data/stocks/cleaned_stock_data.csv'
DESTINATION_FOLDER = 'training-data/stocks/processed_data/'

# Initialize Cloud Storage client
storage_client = storage.Client()

# Function to load CSV data from GCS
def load_data_from_gcs(bucket_name, file_path):
    logging.info(f"Attempting to load data from GCS: bucket={bucket_name}, file_path={file_path}")
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        data = blob.download_as_text()
        df = pd.read_csv(io.StringIO(data))
        logging.info("Data loaded successfully.")
        logging.info(f"Sample data:\n{df.head()}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from GCS: {e}")
        raise

# Function to create lagged features and a 7-day rolling window for each stock
def create_features_by_stock(df, lags=5, rolling_window=7):
    logging.info("Creating lagged features and rolling window within each stock.")
    try:
        # Sort by stock symbol and date within each stock group
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values(['symbol', 'trade_date'])
       
        # Group by stock symbol and apply lagged features within each group
        lagged_data = []
        for symbol, group in df.groupby('symbol'):
            # Creating lagged columns for each variable within each stock
            for lag in range(1, lags + 1):
                group[f'open_lag_{lag}'] = group['open'].shift(lag)
                group[f'high_lag_{lag}'] = group['high'].shift(lag)
                group[f'low_lag_{lag}'] = group['low'].shift(lag)
                group[f'close_lag_{lag}'] = group['close'].shift(lag)
                group[f'volume_lag_{lag}'] = group['volume'].shift(lag)
           
            # Adding rolling window features within each stock
            group['close_7d_avg'] = group['close'].rolling(window=rolling_window).mean()
            group['volume_7d_avg'] = group['volume'].rolling(window=rolling_window).mean()
           
            # Append processed group
            lagged_data.append(group)
       
        # Concatenate all processed groups back into a single DataFrame
        df = pd.concat(lagged_data).dropna()
        logging.info("Features created successfully for each stock.")
        logging.info(f"Sample processed data:\n{df.head()}")
        return df
    except Exception as e:
        logging.error(f"Error creating features: {e}")
        raise

# Function to split the data
def split_data(df):
    logging.info("Splitting data into train, test, and validation sets.")
    try:
        train, temp = train_test_split(df, test_size=0.2, random_state=42)
        test, validation = train_test_split(temp, test_size=0.5, random_state=42)
        logging.info("Data split successfully.")
        return train, test, validation
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

# Function to upload DataFrame as CSV to GCS
def upload_to_gcs(df, bucket_name, destination_path):
    logging.info(f"Uploading data to GCS at {destination_path}")
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_path)
        blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")
        logging.info(f"Data uploaded to {destination_path} successfully.")
    except Exception as e:
        logging.error(f"Error uploading data to GCS: {e}")
        raise

# Main pipeline function
@functions_framework.http
def stock_data_pipeline_http(request):
    try:
        logging.info("Starting stock data pipeline.")

        # Step 1: Load data
        data = load_data_from_gcs(BUCKET_NAME, SOURCE_FILE_PATH)

        # Step 2: Create features within each stock
        processed_data = create_features_by_stock(data)

        # Step 3: Split data
        train_data, test_data, validation_data = split_data(processed_data)

        # Step 4: Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Step 5: Upload each split to GCS
        upload_to_gcs(train_data, BUCKET_NAME, f"{DESTINATION_FOLDER}train_stock_data_{timestamp}.csv")
        upload_to_gcs(test_data, BUCKET_NAME, f"{DESTINATION_FOLDER}test_stock_data_{timestamp}.csv")
        upload_to_gcs(validation_data, BUCKET_NAME, f"{DESTINATION_FOLDER}validation_stock_data_{timestamp}.csv")

        logging.info("Data processing and upload completed successfully.")
        return {"message": "Data processed and uploaded successfully"}, 200

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        return {"error": str(e)}, 500