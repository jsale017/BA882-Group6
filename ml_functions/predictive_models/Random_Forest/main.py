import functions_framework
import pandas as pd
from google.cloud import storage
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import logging
import io
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)

BUCKET_NAME = 'alpha_vatange_vertex_models'
PREDICTION_FOLDER = 'training-data/stocks/predictions/'
PROCESSED_DATA_FOLDER = 'training-data/stocks/processed_data/'

storage_client = storage.Client()

def get_latest_file(bucket_name, prefix, pattern_str):
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    pattern = re.compile(pattern_str)
    
    # Filter blobs to match the file pattern
    filtered_blobs = [blob for blob in blobs if pattern.search(blob.name)]
    
    if not filtered_blobs:
        logging.error(f"No files found in {prefix} matching pattern: {pattern_str}")
        raise FileNotFoundError(f"No files found in {prefix} matching pattern: {pattern_str}")
    
    # Sort by date in filename
    latest_blob = max(
        filtered_blobs,
        key=lambda blob: datetime.strptime(pattern.search(blob.name).group(1), '%Y%m%d%H%M%S')
    )
    
    logging.info(f"Latest file found: {latest_blob.name}")
    return latest_blob.name


def load_data_from_gcs(bucket_name, file_path):
    logging.info(f"Loading data from GCS: bucket={bucket_name}, file_path={file_path}")
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        data = blob.download_as_text()
        df = pd.read_csv(io.StringIO(data))
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data from GCS: {e}")
        raise

def train_and_predict(train_df, valid_df, target_column):
    logging.info(f"Training Random Forest model for {target_column}")
    try:
        feature_columns = [col for col in train_df.columns if col not in ['symbol', 'trade_date', 'close', 'volume']]
        
        # Training and Validation sets
        X_train, y_train = train_df[feature_columns], train_df[target_column]
        X_valid, y_valid = valid_df[feature_columns], valid_df[target_column]
        
        # Initialize and train the Random Forest model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions on the validation set
        predictions = model.predict(X_valid)
        
        # Calculate MSE on the validation set
        mse = mean_squared_error(y_valid, predictions)
        logging.info(f"Model training and prediction completed for {target_column} with MSE: {mse}")

        return predictions, mse
    except Exception as e:
        logging.error(f"Error training model for {target_column}: {e}")
        raise

def upload_predictions_to_gcs(predictions_df, bucket_name, destination_path):
    logging.info(f"Uploading predictions to GCS at {destination_path}")
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_path)
        blob.upload_from_string(predictions_df.to_csv(index=False), content_type="text/csv")
        logging.info(f"Predictions uploaded successfully to {destination_path}.")
    except Exception as e:
        logging.error(f"Error uploading predictions to GCS: {e}")
        raise

@functions_framework.http
def stock_predictions_pipeline_http(request):
    try:
        logging.info("Starting stock predictions pipeline.")

        # Step 1: Load the latest training and validation data files
        latest_train_file_path = get_latest_file(BUCKET_NAME, PROCESSED_DATA_FOLDER, r'train_stock_data_(\d{14})\.csv')
        latest_valid_file_path = get_latest_file(BUCKET_NAME, PROCESSED_DATA_FOLDER, r'validation_stock_data_(\d{14})\.csv')

        train_data = load_data_from_gcs(BUCKET_NAME, latest_train_file_path)
        valid_data = load_data_from_gcs(BUCKET_NAME, latest_valid_file_path)

        # Step 2: Train models and generate predictions on the validation set
        volume_predictions, volume_mse = train_and_predict(train_data, valid_data, target_column='volume')
        close_predictions, close_mse = train_and_predict(train_data, valid_data, target_column='close')

        # Step 3: Combine validation set data and predictions into a DataFrame
        results_df = valid_data[['symbol', 'trade_date']].copy()
        results_df['volume_prediction'] = volume_predictions
        results_df['close_prediction'] = close_predictions

        # Step 4: Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        destination_path = f"{PREDICTION_FOLDER}random_forest_predictions_{timestamp}.csv"

        # Step 5: Upload predictions to GCS
        upload_predictions_to_gcs(results_df, BUCKET_NAME, destination_path)

        logging.info("Prediction and upload completed successfully.")
        return {
            "message": "Predictions generated and uploaded successfully",
            "volume_mse": volume_mse,
            "close_mse": close_mse
        }, 200

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        return {"error": str(e)}, 500
