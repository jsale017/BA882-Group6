import functions_framework
import pandas as pd
from google.cloud import storage, bigquery
from google.cloud.exceptions import NotFound
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import logging
import io
from datetime import datetime
from datetime import timedelta
import re

logging.basicConfig(level=logging.INFO)

# Constants
BUCKET_NAME = 'alpha_vatange_vertex_models'
PREDICTION_FOLDER = 'training-data/stocks/predictions/'
PROCESSED_DATA_FOLDER = 'training-data/stocks/processed_data/'
BIGQUERY_DATASET = 'financial_data'
BIGQUERY_TABLE = 'ML_predictions_random_forest_predictions'

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

def ensure_dataset_exists(client, dataset_name):
    """
    Ensure the BigQuery dataset exists.
    """
    try:
        client.get_dataset(dataset_name)
        logging.info(f"Dataset {dataset_name} already exists.")
    except NotFound:
        logging.info(f"Dataset {dataset_name} not found. Creating dataset...")
        dataset = bigquery.Dataset(f"{client.project}.{dataset_name}")
        client.create_dataset(dataset)
        logging.info(f"Dataset {dataset_name} created successfully.")

def create_table_if_not_exists(client, dataset_name, table_name):
    """
    Automatically create the BigQuery table if it doesn't exist.
    """
    table_id = f"{client.project}.{dataset_name}.{table_name}"
    try:
        client.get_table(table_id)  # Check if table exists
        logging.info(f"Table {table_id} already exists.")
    except NotFound:
        logging.info(f"Table {table_id} not found. Creating table...")
        schema = [
            bigquery.SchemaField("symbol", "STRING"),
            bigquery.SchemaField("trade_date", "DATE"),
            bigquery.SchemaField("volume_prediction", "FLOAT"),
            bigquery.SchemaField("close_prediction", "FLOAT"),
        ]
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table)
        logging.info(f"Table {table_id} created successfully.")

def save_predictions_to_bigquery(predictions_df, dataset_name, table_name):
    """
    Save predictions DataFrame to a BigQuery table. Overwrites predictions for the same dates.
    """
    logging.info(f"Saving predictions to BigQuery: {dataset_name}.{table_name}")
    try:
        client = bigquery.Client()

        # Ensure dataset and table exist
        ensure_dataset_exists(client, dataset_name)
        create_table_if_not_exists(client, dataset_name, table_name)

        # Set table reference
        table_id = f"{client.project}.{dataset_name}.{table_name}"

        # Convert trade_date to datetime and ensure it is valid
        predictions_df["trade_date"] = pd.to_datetime(predictions_df["trade_date"], errors="coerce")
        if predictions_df["trade_date"].isnull().any():
            raise ValueError("Invalid dates found in the 'trade_date' column.")

        # Get the unique dates and symbols from the new predictions
        unique_dates = predictions_df["trade_date"].dt.date.unique()  # Convert to DATE type
        unique_symbols = predictions_df["symbol"].unique()

        # Construct the DELETE query
        delete_query = f"""
        DELETE FROM `{table_id}`
        WHERE trade_date IN UNNEST([{', '.join(f'DATE("{date}")' for date in unique_dates)}])
        AND symbol IN UNNEST([{', '.join(f'"{symbol}"' for symbol in unique_symbols)}])
        """
        logging.info(f"Running delete query: {delete_query}")
        client.query(delete_query).result()
        logging.info("Existing predictions deleted successfully.")

        # Define job configuration for BigQuery upload
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND
        )

        # Upload DataFrame to BigQuery
        job = client.load_table_from_dataframe(
            predictions_df, table_id, job_config=job_config
        )
        job.result()
        logging.info(f"Predictions saved successfully to BigQuery: {table_id}")
    except Exception as e:
        logging.error(f"Error saving predictions to BigQuery: {e}")
        raise

@functions_framework.http
def stock_predictions_pipeline_http(request):
    try:
        logging.info("Starting stock predictions pipeline.")

        # Step 1: Load the latest training and validation data files
        latest_train_file_path = get_latest_file(BUCKET_NAME, PROCESSED_DATA_FOLDER, r'train_stock_data_(\d{14})\.csv')

        train_data = load_data_from_gcs(BUCKET_NAME, latest_train_file_path)

        # Step 2: Ensure trade_date is in datetime format and filter data up to yesterday
        train_data["trade_date"] = pd.to_datetime(train_data["trade_date"])
        yesterday = datetime.now() - timedelta(days=1)
        train_data = train_data[train_data["trade_date"] <= pd.Timestamp(yesterday)]

        # Step 3: Prepare data for predictions
        stocks = train_data["symbol"].unique()
        prediction_results = []

        for stock in stocks:
            stock_data = train_data[train_data["symbol"] == stock]

            # Ensure data is sorted by date
            stock_data = stock_data.sort_values(by="trade_date")

            # Use all available data to train
            feature_columns = [col for col in stock_data.columns if col not in ['symbol', 'trade_date', 'close', 'volume']]

            X_train = stock_data[feature_columns]
            y_volume = stock_data["volume"]
            y_close = stock_data["close"]

            # Initialize and train models
            volume_model = RandomForestRegressor(random_state=42)
            volume_model.fit(X_train, y_volume)

            close_model = RandomForestRegressor(random_state=42)
            close_model.fit(X_train, y_close)

            # Create features for the next 5 days
            last_features = stock_data[feature_columns].iloc[-1].values.reshape(1, -1)
            for i in range(1, 6):
                next_date = yesterday.date() + timedelta(days=i)
                volume_prediction = volume_model.predict(last_features)[0]
                close_prediction = close_model.predict(last_features)[0]

                # Append results for the stock
                prediction_results.append({
                    "symbol": stock,
                    "trade_date": next_date,
                    "volume_prediction": volume_prediction,
                    "close_prediction": close_prediction
                })

        # Step 4: Create a DataFrame for the predictions
        results_df = pd.DataFrame(prediction_results)

        # Step 5: Upload predictions to BigQuery
        save_predictions_to_bigquery(results_df, BIGQUERY_DATASET, BIGQUERY_TABLE)

        logging.info("Prediction and upload completed successfully.")
        return {
            "message": "5-day predictions generated and uploaded successfully.",
            "predictions": results_df.to_dict(orient="records")
        }, 200

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        return {"error": str(e)}, 500
