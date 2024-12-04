import functions_framework
import pandas as pd
from google.cloud import storage, bigquery
from google.cloud.exceptions import NotFound
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import logging
import io
from datetime import datetime, timedelta
import re

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
BUCKET_NAME = 'alpha_vatange_vertex_models'
PROCESSED_DATA_FOLDER = 'training-data/stocks/processed_data/'
BIGQUERY_DATASET = 'financial_data'
BIGQUERY_TABLE = 'ML_predictions_random_forest_predictions'

# Initialize storage client
storage_client = storage.Client()

def get_latest_file(bucket_name, prefix, pattern_str):
    """Get the latest file from GCS based on a timestamp pattern."""
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    pattern = re.compile(pattern_str)

    filtered_blobs = [blob for blob in blobs if pattern.search(blob.name)]
    if not filtered_blobs:
        logging.error(f"No files found in {prefix} matching pattern: {pattern_str}")
        raise FileNotFoundError(f"No files found in {prefix} matching pattern: {pattern_str}")

    latest_blob = max(
        filtered_blobs,
        key=lambda blob: datetime.strptime(pattern.search(blob.name).group(1), '%Y%m%d%H%M%S')
    )
    logging.info(f"Latest file found: {latest_blob.name}")
    return latest_blob.name

def load_data_from_gcs(bucket_name, file_path):
    """Load CSV data from GCS."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        data = blob.download_as_text()
        return pd.read_csv(io.StringIO(data))
    except Exception as e:
        logging.error(f"Error loading data from GCS: {e}")
        raise

def ensure_dataset_and_table(client, dataset_name, table_name):
    """Ensure BigQuery dataset and table exist."""
    dataset_ref = bigquery.Dataset(client.dataset(dataset_name))
    try:
        client.get_dataset(dataset_ref)
        logging.info(f"Dataset {dataset_name} exists.")
    except NotFound:
        logging.info(f"Dataset {dataset_name} not found. Creating...")
        client.create_dataset(dataset_ref)
        logging.info(f"Dataset {dataset_name} created.")

    table_id = f"{client.project}.{dataset_name}.{table_name}"
    try:
        client.get_table(table_id)
        logging.info(f"Table {table_id} exists.")
    except NotFound:
        schema = [
            bigquery.SchemaField("symbol", "STRING"),
            bigquery.SchemaField("trade_date", "DATE"),
            bigquery.SchemaField("volume_prediction", "FLOAT"),
            bigquery.SchemaField("close_prediction", "FLOAT"),
        ]
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table)
        logging.info(f"Table {table_id} created.")

def save_predictions_to_bigquery(predictions_df, dataset_name, table_name):
    """Save predictions DataFrame to BigQuery."""
    try:
        client = bigquery.Client()
        ensure_dataset_and_table(client, dataset_name, table_name)

        table_id = f"{client.project}.{dataset_name}.{table_name}"
        predictions_df["trade_date"] = pd.to_datetime(predictions_df["trade_date"], errors="coerce")
        if predictions_df["trade_date"].isnull().any():
            raise ValueError("Invalid dates found in 'trade_date' column.")

        # Define the job configuration
        job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
        job = client.load_table_from_dataframe(predictions_df, table_id, job_config=job_config)
        job.result()
        logging.info(f"Predictions successfully uploaded to {table_id}")
    except Exception as e:
        logging.error(f"Error saving predictions to BigQuery: {e}")
        raise

@functions_framework.http
def stock_predictions_pipeline_http(request):
    """Cloud Function entry point for stock predictions."""
    try:
        logging.info("Starting stock predictions pipeline.")

        # Step 1: Load the latest training data
        latest_train_file_path = get_latest_file(BUCKET_NAME, PROCESSED_DATA_FOLDER, r'train_stock_data_(\d{14})\.csv')
        train_data = load_data_from_gcs(BUCKET_NAME, latest_train_file_path)

        # Step 2: Filter data up to yesterday
        train_data["trade_date"] = pd.to_datetime(train_data["trade_date"])
        yesterday = datetime.now() - timedelta(days=1)
        train_data = train_data[train_data["trade_date"] <= pd.Timestamp(yesterday)]

        # Step 3: Prepare and train models
        stocks = train_data["symbol"].unique()
        prediction_results = []

        for stock in stocks:
            stock_data = train_data[train_data["symbol"] == stock].sort_values(by="trade_date")
            feature_columns = [col for col in stock_data.columns if col not in ['symbol', 'trade_date', 'close', 'volume']]

            X_train = stock_data[feature_columns]
            y_volume = stock_data["volume"]
            y_close = stock_data["close"]

            # Train Random Forest models
            volume_model = RandomForestRegressor(random_state=42)
            volume_model.fit(X_train, y_volume)

            close_model = RandomForestRegressor(random_state=42)
            close_model.fit(X_train, y_close)

            # Predict for the next 5 days
            last_features = stock_data[feature_columns].iloc[-1].values.reshape(1, -1)
            for i in range(1, 6):
                next_date = yesterday.date() + timedelta(days=i)
                volume_prediction = volume_model.predict(last_features)[0]
                close_prediction = close_model.predict(last_features)[0]

                prediction_results.append({
                    "symbol": stock,
                    "trade_date": next_date,
                    "volume_prediction": volume_prediction,
                    "close_prediction": close_prediction
                })

        # Step 4: Save predictions to BigQuery
        results_df = pd.DataFrame(prediction_results)
        save_predictions_to_bigquery(results_df, BIGQUERY_DATASET, BIGQUERY_TABLE)

        logging.info("Prediction pipeline completed successfully.")
        return {
            "message": "5-day predictions generated and uploaded successfully.",
            "predictions": results_df.to_dict(orient="records")
        }, 200

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        return {"error": str(e)}, 500
