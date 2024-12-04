import functions_framework
import pandas as pd
from google.cloud import storage, bigquery
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import logging
from datetime import datetime, timedelta
import io
import re

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
BUCKET_NAME = 'alpha_vatange_vertex_models'
PROCESSED_DATA_FOLDER = 'training-data/stocks/processed_data/'
BIGQUERY_DATASET = 'financial_data'
BIGQUERY_TABLE = 'ML_predictions_xgboost_predictions'

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

def save_predictions_to_bigquery(predictions_df, dataset_name, table_name):
    """Save predictions to BigQuery."""
    try:
        client = bigquery.Client()
        table_id = f"{client.project}.{dataset_name}.{table_name}"

        predictions_df["trade_date"] = pd.to_datetime(predictions_df["trade_date"], errors="coerce")
        if predictions_df["trade_date"].isnull().any():
            raise ValueError("Invalid dates found in 'trade_date' column.")

        job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
        client.load_table_from_dataframe(predictions_df, table_id, job_config=job_config).result()
    except Exception as e:
        logging.error(f"Error saving predictions to BigQuery: {e}")
        raise

@functions_framework.http
def stock_predictions_pipeline_http(request):
    """Cloud Function entry point for stock predictions."""
    try:
        # Get the latest training data file
        latest_train_file_path = get_latest_file(BUCKET_NAME, PROCESSED_DATA_FOLDER, r'train_stock_data_(\d{14})\.csv')
        train_data = load_data_from_gcs(BUCKET_NAME, latest_train_file_path)

        # Preprocess training data
        train_data["trade_date"] = pd.to_datetime(train_data["trade_date"])
        yesterday = datetime.now() - timedelta(days=1)
        train_data = train_data[train_data["trade_date"] <= pd.Timestamp(yesterday)]

        stocks = train_data["symbol"].unique()
        prediction_results = []

        for stock in stocks:
            stock_data = train_data[train_data["symbol"] == stock]
            stock_data = stock_data.sort_values(by="trade_date")

            feature_columns = [col for col in stock_data.columns if col not in ['symbol', 'trade_date', 'close', 'volume']]
            X_train = stock_data[feature_columns]
            y_volume = stock_data["volume"]
            y_close = stock_data["close"]

            # Scale features
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(stock_data[feature_columns])

            # Train models
            volume_model = XGBRegressor(random_state=42, use_label_encoder=False)
            volume_model.fit(scaled_features, y_volume)

            close_model = XGBRegressor(random_state=42, use_label_encoder=False)
            close_model.fit(scaled_features, y_close)

            # Start predictions with the last feature vector
            last_features = scaled_features[-1].reshape(1, -1)  # Ensure last_features has the correct shape

            for i in range(1, 6):
                next_date = (yesterday + timedelta(days=i)).strftime('%Y-%m-%d')

                # Predict volume and close
                volume_prediction = volume_model.predict(last_features)[0]
                close_prediction = close_model.predict(last_features)[0]

                # Append the prediction
                prediction_results.append({
                    "symbol": stock,
                    "trade_date": next_date,
                    "volume_prediction": volume_prediction,
                    "close_prediction": close_prediction
                })

                # Update last_features for the next prediction
                updated_features = last_features.flatten()  # Flatten to update specific values

                # Update `close` and `volume` predictions in the scaled range
                updated_features[-2] = scaler.transform([[*last_features.flatten()[:-2], close_prediction, 0]])[0][-2]  # Update close
                updated_features[-1] = scaler.transform([[*last_features.flatten()[:-2], 0, volume_prediction]])[0][-1]  # Update volume

                last_features = updated_features.reshape(1, -1)  # Reshape back to original dimensions

        # Save predictions to BigQuery
        results_df = pd.DataFrame(prediction_results)
        save_predictions_to_bigquery(results_df, BIGQUERY_DATASET, BIGQUERY_TABLE)

        return {
            "message": "5-day predictions generated and uploaded successfully.",
            "predictions": results_df.to_dict(orient="records")
        }, 200

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        return {"error": str(e)}, 500
