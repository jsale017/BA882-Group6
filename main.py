import functions_framework
import pandas as pd
from google.cloud import storage
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import logging
import io
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)

# Constants
BUCKET_NAME = 'alpha_vatange_vertex_models'
TRAIN_DATA_PATH = 'training-data/stocks/processed_data/train_stock_data_20241108193729.csv'
TEST_DATA_PATH = 'training-data/stocks/processed_data/test_stock_data_20241108193729.csv'
PREDICTION_FOLDER = 'training-data/stocks/predictions/'


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

# Function to train XGBoost models and make predictions
def train_and_predict(train_df, test_df, target_column):
    logging.info(f"Training XGBoost model for target: {target_column}")
    try:
        # Identify feature columns by excluding non-numerical and target columns
        feature_columns = [col for col in train_df.columns if col not in ['symbol', 'trade_date', 'close', 'volume']]

        # Split data into features and target
        X_train, y_train = train_df[feature_columns], train_df[target_column]
        X_test, y_test = test_df[feature_columns], test_df[target_column]

        # Initialize and train the XGBoost model
        model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=100)
        model.fit(X_train, y_train)
        
        # Predict on the testing data
        predictions = model.predict(X_test)
        
        # Calculate MSE
        mse = mean_squared_error(y_test, predictions)
        logging.info(f"Model MSE for {target_column}: {mse}")

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

        # Step 1: Load train and test data
        train_data = load_data_from_gcs(BUCKET_NAME, TRAIN_DATA_PATH)
        test_data = load_data_from_gcs(BUCKET_NAME, TEST_DATA_PATH)

        # Step 2: Train models and generate predictions on the test set
        volume_predictions, volume_mse = train_and_predict(train_data, test_data, target_column='volume')
        close_predictions, close_mse = train_and_predict(train_data, test_data, target_column='close')

        # Step 3: Combine test set data and predictions into a DataFrame
        results_df = test_data[['symbol', 'trade_date']].copy()
        results_df['volume_prediction'] = volume_predictions
        results_df['close_prediction'] = close_predictions
        results_df['volume_mse'] = volume_mse
        results_df['close_mse'] = close_mse

        # Step 4: Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        destination_path = f"{PREDICTION_FOLDER}stock_predictions_{timestamp}.csv"

        # Step 5: Upload predictions to GCS
        upload_predictions_to_gcs(results_df, BUCKET_NAME, destination_path)

        logging.info("Prediction and upload completed successfully.")
        return {"message": "Predictions generated and uploaded successfully", "volume_mse": volume_mse, "close_mse": close_mse}, 200

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        return {"error": str(e)}, 500