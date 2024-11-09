from google.cloud import storage
import pandas as pd
import io
from flask import jsonify
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout
from sklearn.metrics import mean_squared_error

# Define your bucket name and file paths
BUCKET_NAME = 'alpha_vatange_vertex_models'
FOLDER_PATH = 'training-data/stocks/'
OUTPUT_FOLDER_PATH = 'training-data/stocks/predictions/'  # Folder for storing predictions

# Initialize the Google Cloud Storage client
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

def read_csv_from_gcs_and_predict(request):
    file_name = request.args.get("file_name", "cleaned_stock_data.csv")  # Get file name from query parameter
    try:
        # Load data from GCS
        blob = bucket.blob(f"{FOLDER_PATH}{file_name}")
        data = blob.download_as_text()
        df = pd.read_csv(io.StringIO(data))

        # Preprocess data
        df, scaler = preprocess_data(df)
        
        # Prepare data for sequence creation
        data_values = df[['open', 'high', 'low', 'close', 'volume']].values
        X_train, X_test, y_train, y_test = create_sequences(data_values, sequence_length=60)

        # Train and evaluate LSTM model
        lstm_model, _ = build_and_train_model(X_train, y_train, X_test, y_test, model_type='LSTM')
        #lstm_predictions_rescaled, y_test_rescaled, LSTM_rmse_close, LSTM_rmse_volume = make_predictions_and_evaluate(lstm_model, X_test, y_test, scaler)
        #LSTM_df = create_results_dataframe(df, y_test_rescaled, lstm_predictions_rescaled, y_test)
        lstm_predictions_rescaled, y_test_rescaled, LSTM_rmse_close, LSTM_rmse_volume = make_predictions_and_evaluate(lstm_model, X_test, y_test, scaler)
        LSTM_df = create_results_dataframe(df, y_test_rescaled, lstm_predictions_rescaled, y_test, LSTM_rmse_close, LSTM_rmse_volume)

        # Train and evaluate RNN model
        rnn_model, _ = build_and_train_model(X_train, y_train, X_test, y_test, model_type='RNN')
        #rnn_predictions_rescaled, y_test_rescaled, RNN_rmse_close, RNN_rmse_volume = make_predictions_and_evaluate(rnn_model, X_test, y_test, scaler)
        #RNN_df = create_results_dataframe(df, y_test_rescaled, rnn_predictions_rescaled, y_test)
        rnn_predictions_rescaled, y_test_rescaled, RNN_rmse_close, RNN_rmse_volume = make_predictions_and_evaluate(rnn_model, X_test, y_test, scaler)
        RNN_df = create_results_dataframe(df, y_test_rescaled, rnn_predictions_rescaled, y_test, RNN_rmse_close, RNN_rmse_volume)

        # RMSE comparison DataFrame
        rmse_df = pd.DataFrame({
            'Model': ['LSTM', 'RNN'],
            'RMSE_Close': [LSTM_rmse_close, RNN_rmse_close],
            'RMSE_Volume': [LSTM_rmse_volume, RNN_rmse_volume]
        })

        # Store DataFrames in GCS
        store_dataframe_in_gcs(LSTM_df, f"{OUTPUT_FOLDER_PATH}LSTM_predictions.csv")
        store_dataframe_in_gcs(RNN_df, f"{OUTPUT_FOLDER_PATH}RNN_predictions.csv")
        # store_dataframe_in_gcs(rmse_df, f"{OUTPUT_FOLDER_PATH}RMSE_comparison.csv")

        # Create response message
        response_message = {
            "message": "Data loaded, models trained, predictions made, and results stored in GCS successfully",
            "RMSE_Comparison": rmse_df.to_dict(orient='records')
        }
        return jsonify(response_message), 200

    except Exception as e:
        error_message = f"Error processing {file_name}: {e}"
        print(error_message)
        return jsonify({"error": error_message}), 500

def preprocess_data(df, symbol='AAPL'):
    df = df[df['symbol'] == symbol]
    df = df.drop_duplicates(subset=['symbol', 'trade_date'])
    df = df.sort_values(by='trade_date')
    scaler = MinMaxScaler()
    df[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])
    return df, scaler

def create_sequences(data, sequence_length=60, split_ratio=0.8):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, [3, 4]])  # 3 is close, 4 is volume
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
    model.add(Dense(units=2))  # 2 outputs: close price and volume

    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return model, history

def make_predictions_and_evaluate(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(np.hstack([np.zeros((predictions.shape[0], 3)), predictions]))
    y_test_rescaled = scaler.inverse_transform(np.hstack([np.zeros((y_test.shape[0], 3)), y_test]))

    actual_close_volume = y_test_rescaled[:, 3:5]
    predicted_close_volume = predictions_rescaled[:, 3:5]
    rmse_close = mean_squared_error(actual_close_volume[:, 0], predicted_close_volume[:, 0], squared=False)
    rmse_volume = mean_squared_error(actual_close_volume[:, 1], predicted_close_volume[:, 1], squared=False)

    return predictions_rescaled, y_test_rescaled, rmse_close, rmse_volume


def create_results_dataframe(stock_df, y_test_rescaled, predictions_rescaled, y_test, rmse_close, rmse_volume):
    actual_close_volume = y_test_rescaled[:, 3:5]
    predicted_close_volume = predictions_rescaled[:, 3:5]
    dates = stock_df['trade_date'].iloc[-len(y_test):].values

    # Create the results DataFrame
    results_df = pd.DataFrame({
        'Date': dates,
        'Open': stock_df['open'].iloc[-len(y_test):].values,
        'High': stock_df['high'].iloc[-len(y_test):].values,
        'Low': stock_df['low'].iloc[-len(y_test):].values,
        'Actual Close': actual_close_volume[:, 0],
        'Predicted Close': predicted_close_volume[:, 0],
        'Actual Volume': actual_close_volume[:, 1],
        'Predicted Volume': predicted_close_volume[:, 1],
        'RMSE_Close': rmse_close,         # Add RMSE values as new columns
        'RMSE_Volume': rmse_volume        # These will be the same value for each row
    })
    return results_df

def store_dataframe_in_gcs(df, destination_path):
    """Save a DataFrame as a CSV file in Google Cloud Storage."""
    blob = bucket.blob(destination_path)
    csv_data = df.to_csv(index=False)
    blob.upload_from_string(csv_data, content_type='text/csv')
