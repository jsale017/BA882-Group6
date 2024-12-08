from google.cloud import bigquery, storage
import pandas as pd
import io
from flask import jsonify
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout
from sklearn.metrics import mean_squared_error

BUCKET_NAME = 'alpha_vatange_vertex_models'
# Storage path for Big Query
BIGQUERY_PROJECT_ID = 'finnhub-pipeline-ba882' 
BIGQUERY_DATASET_ID = 'financial_data'
# Storage path for GCS
FOLDER_PATH = 'training-data/stocks/'
OUTPUT_FOLDER_PATH = 'training-data/stocks/predictions/'  

# Initialize Google Cloud clients
storage_client = storage.Client()
bigquery_client = bigquery.Client(project=BIGQUERY_PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)

def save_dataframe_to_bigquery(df, table_id):
    """Save a DataFrame to BigQuery after ensuring schema compatibility."""
    if 'trade_date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
        if df['trade_date'].isnull().any():
            raise ValueError("Some entries in 'trade_date' could not be converted to datetime format.")

    table_ref = bigquery_client.dataset(table_id.split(".")[0]).table(table_id.split(".")[1])
    try:
        bigquery_client.get_table(table_ref)
    except Exception as e:
        if "Not found" in str(e):
            print(f"Table {table_id} does not exist. Creating it...")
            schema = [
                bigquery.SchemaField("symbol", "STRING"),
                bigquery.SchemaField("trade_date", "DATE"),
                bigquery.SchemaField("volume_prediction", "FLOAT"),
                bigquery.SchemaField("close_prediction", "FLOAT"),
            ]
            table = bigquery.Table(table_ref, schema=schema)
            bigquery_client.create_table(table)
            print(f"Table {table_id} created successfully.")
    
    query = f"DELETE FROM `{table_id}` WHERE TRUE"
    bigquery_client.query(query).result()
    print(f"Data cleared from table {table_id}.")

    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
    job = bigquery_client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    print(f"Data successfully saved to BigQuery table {table_id}.")

def read_csv_from_gcs_and_predict(request):
    file_name = request.args.get("file_name", "cleaned_stock_data.csv")
    try:
        blob = storage_client.bucket(BUCKET_NAME).blob(f"training-data/stocks/{file_name}")
        data = blob.download_as_text()
        df = pd.read_csv(io.StringIO(data))

        lstm_results = []
        rnn_results = []

        symbols = df['symbol'].unique()
        for symbol in symbols:
            df_symbol, scaler = preprocess_data(df, symbol)
            data_values = df_symbol[['open', 'high', 'low', 'close', 'volume']].values
            X_train, X_test, y_train, y_test = create_sequences(data_values, sequence_length=60)

            lstm_model, _ = build_and_train_model(X_train, y_train, X_test, y_test, model_type='LSTM')
            lstm_predictions_rescaled, y_test_rescaled, LSTM_mse_close, LSTM_mse_volume = make_predictions_and_evaluate(
                lstm_model, X_test, y_test, scaler)
            lstm_future_predictions = make_future_predictions(lstm_model, X_test[-1], scaler, days=5)
            LSTM_df = create_results_dataframe(
                df_symbol, y_test_rescaled, lstm_predictions_rescaled, y_test, LSTM_mse_close, LSTM_mse_volume, lstm_future_predictions, 'LSTM', symbol
            )
            lstm_results.append(LSTM_df)

            rnn_model, _ = build_and_train_model(X_train, y_train, X_test, y_test, model_type='RNN')
            rnn_predictions_rescaled, y_test_rescaled, RNN_mse_close, RNN_mse_volume = make_predictions_and_evaluate(
                rnn_model, X_test, y_test, scaler)
            rnn_future_predictions = make_future_predictions(rnn_model, X_test[-1], scaler, days=5)
            RNN_df = create_results_dataframe(
                df_symbol, y_test_rescaled, rnn_predictions_rescaled, y_test, RNN_mse_close, RNN_mse_volume, rnn_future_predictions, 'RNN', symbol
            )
            rnn_results.append(RNN_df)

        combined_lstm_results = pd.concat(lstm_results, ignore_index=True)
        combined_rnn_results = pd.concat(rnn_results, ignore_index=True)

        save_dataframe_to_bigquery(combined_lstm_results, f"{BIGQUERY_DATASET_ID}.ML_LSTM_predictions")
        save_dataframe_to_bigquery(combined_rnn_results, f"{BIGQUERY_DATASET_ID}.ML_RNN_predictions")

        store_dataframe_in_gcs(combined_lstm_results, f"{OUTPUT_FOLDER_PATH}ML_LSTM_predictions.csv")
        store_dataframe_in_gcs(combined_rnn_results, f"{OUTPUT_FOLDER_PATH}ML_RNN_predictions.csv")

        response_message = {
            "message": "Data loaded, models trained, predictions made, and results stored in BigQuery successfully"
        }
        return jsonify(response_message), 200

    except Exception as e:
        error_message = f"Error processing {file_name}: {e}"
        print(error_message)
        return jsonify({"error": error_message}), 500

def preprocess_data(df, symbol):
    df = df[df['symbol'] == symbol].drop_duplicates(subset=['symbol', 'trade_date']).sort_values(by=['symbol', 'trade_date'])
    scaler = MinMaxScaler()
    df[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])
    return df, scaler

def create_sequences(data, sequence_length=60, split_ratio=0.8):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, [3, 4]])
        y.append(data[i + sequence_length, [3, 4]])

    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * split_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test

def build_and_train_model(X_train, y_train, X_test, y_test, model_type='LSTM', epochs=5, batch_size=32):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
    elif model_type == 'RNN':
        model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=2))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return model, None

def make_predictions_and_evaluate(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(np.hstack([np.zeros((predictions.shape[0], 3)), predictions]))
    y_test_rescaled = scaler.inverse_transform(np.hstack([np.zeros((y_test.shape[0], 3)), y_test]))

    actual_close_volume = y_test_rescaled[:, 3:5]
    predicted_close_volume = predictions_rescaled[:, 3:5]
    mse_close = mean_squared_error(actual_close_volume[:, 0], predicted_close_volume[:, 0])
    mse_volume = mean_squared_error(actual_close_volume[:, 1], predicted_close_volume[:, 1])

    return predictions_rescaled, y_test_rescaled, mse_close, mse_volume

def make_future_predictions(model, last_sequence, scaler, days=5):
    predictions = []
    input_sequence = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])

    for _ in range(days):
        predicted = model.predict(input_sequence)
        predictions.append(predicted[0])
        input_sequence = np.roll(input_sequence, -1, axis=1)
        input_sequence[0, -1, :] = predicted

    predictions_rescaled = scaler.inverse_transform(np.hstack([np.zeros((days, 3)), np.array(predictions)]))
    return predictions_rescaled[:, 3:5]

def create_results_dataframe(stock_df, y_test_rescaled, predictions_rescaled, y_test, mse_close, mse_volume, future_predictions, model_type, symbol):
    actual_close_volume = y_test_rescaled[:, 3:5]
    predicted_close_volume = predictions_rescaled[:, 3:5]
    dates = stock_df['trade_date'].iloc[-len(y_test):].values

    dates = pd.to_datetime(dates, errors='coerce')
    last_date = max(pd.to_datetime(dates[-1]), pd.Timestamp.now().normalize())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_predictions), freq='D')
    future_df = pd.DataFrame({
        'symbol': [symbol] * len(future_dates),
        'trade_date': future_dates,
        'volume_prediction': future_predictions[:, 1],
        'close_prediction': future_predictions[:, 0]
    })

    return future_df

def store_dataframe_in_gcs(df, destination_path):
    """Save a DataFrame as a CSV file in Google Cloud Storage."""
    blob = bucket.blob(destination_path)
    csv_data = df.to_csv(index=False)
    blob.upload_from_string(csv_data, content_type='text/csv')
