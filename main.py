import functions_framework
import pandas as pd
from google.cloud import storage
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import logging
import io
from datetime import datetime, timedelta
import re

# Set up logging
logging.basicConfig(level=logging.INFO)

# Constants
BUCKET_NAME = 'alpha_vatange_vertex_models'
PREDICTION_FOLDER = 'training-data/stocks/predictions/'
PROCESSED_DATA_FOLDER = 'training-data/stocks/processed_data/'
PREDICTION_HORIZON = 5 

# Initialize Cloud Storage client
storage_client = storage.Client()

# Function to find the latest training data file
def get_latest_file(bucket_name, prefix):
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    # Filter blobs to match the file pattern and sort by date in filename
    pattern = re.compile(r'train_stock_data_(\d{14})\.csv')
    latest_blob = max(
        (blob for blob in blobs if pattern.search(blob.name)),
        key=lambda blob: datetime.strptime(pattern.search(blob.name).group(1), '%Y%m%d%H%M%S')
    )
    
    logging.info(f"Latest file found: {latest_blob.name}")
    return latest_blob.name

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

# Function to generate future test data for prediction
def create_future_test_data(train_df, horizon):
    logging.info("Creating future test data based on the latest date in training data.")
    last_date = pd.to_datetime(train_df['trade_date'].max())
    symbols = train_df['symbol'].unique()

    # Create a DataFrame for future dates
    future_dates = pd.DataFrame({
        'trade_date': [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
    })

    # Repeat for each symbol in the dataset
    future_test_data = pd.concat([
        future_dates.assign(symbol=symbol) for symbol in symbols
    ]).reset_index(drop=True)

    # Add placeholder columns for the features expected by the model
    feature_columns = [col for col in train_df.columns if col not in ['symbol', 'trade_date', 'close', 'volume']]
    for col in feature_columns:
        future_test_data[col] = 0  # Placeholder values

    logging.info("Future test data created successfully.")
    logging.info(f"Future test data sample:\n{future_test_data.head()}")
    return future_test_data

# Function to train XGBoost models and make predictions
def train_and_predict(train_df, test_df, target_column):
    logging.info(f"Training XGBoost model for target: {target_column}")
    try:
        # Identify feature columns by excluding non-numerical and target columns
        feature_columns = [col for col in train_df.columns if col not in ['symbol', 'trade_date', 'close', 'volume']]

        # Split data into features and target
        X_train, y_train = train_df[feature_columns], train_df[target_column]
        X_test = test_df[feature_columns]

        # Initialize and train the XGBoost model
        model = XGBRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        # Predict on the testing data
        predictions = model.predict(X_test)
        
        # Calculate MSE if there’s actual data available
        mse = mean_squared_error(y_train[-len(predictions):], predictions) if target_column in ['volume', 'close'] else None
        logging.info(f"Model training and prediction completed for {target_column}. MSE: {mse}")

        return predictions, mse
    except Exception as e:
        logging.error(f"Error training model for {target_column}: {e}")
        raise

# Function to upload predictions as CSV to GCS
def upload_predictions_to_gcs(predictions_df, bucket_name, destination_path):
    logging.info(f"Uploading predictions to GCS at {destination_path}")
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_path)
        blob.upload_from_string(predictions_df.to_csv(index=False), content_type="text/csv")
        logging.info(f"Predictions uploaded to {destination_path} successfully.")
    except Exception as e:
        logging.error(f"Error uploading predictions to GCS: {e}")
        raise

# Main pipeline function
@functions_framework.http
def stock_predictions_pipeline_http(request):
    try:
        logging.info("Starting stock predictions pipeline.")

        # Step 1: Get the latest training data file
        latest_file_path = get_latest_file(BUCKET_NAME, PROCESSED_DATA_FOLDER)

        # Step 2: Load the latest training data
        train_data = load_data_from_gcs(BUCKET_NAME, latest_file_path)

        # Step 3: Generate future test data
        test_data = create_future_test_data(train_data, PREDICTION_HORIZON)

        # Step 4: Train models and generate predictions on the future test set
        volume_predictions, volume_mse = train_and_predict(train_data, test_data, target_column='volume')
        close_predictions, close_mse = train_and_predict(train_data, test_data, target_column='close')

        # Step 5: Combine future test set data and predictions into a DataFrame
        results_df = test_data[['symbol', 'trade_date']].copy()
        results_df['volume_prediction'] = volume_predictions
        results_df['close_prediction'] = close_predictions

        # Step 6: Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        destination_path = f"{PREDICTION_FOLDER}xgboost_predictions_{timestamp}.csv"

        # Step 7: Upload predictions to GCS
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
