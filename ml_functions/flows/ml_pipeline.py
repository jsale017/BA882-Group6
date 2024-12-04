# imports
import requests
import json
from prefect import flow, task

# helper function
def invoke_gcf(url: str, payload: dict):
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

@task(retries=2)
def clean_and_prepare_stock_data():
    """Clean and prepare stock data"""
    url = "https://us-central1-finnhub-pipeline-ba882.cloudfunctions.net/clean_and_prepare_stock_data"
    resp = invoke_gcf(url, payload={})
    return resp

@task(retries=2)
def rolling_window():
    """Apply a rolling window to the stock data"""
    url = "https://us-central1-finnhub-pipeline-ba882.cloudfunctions.net/stock_data_pipeline_http"
    resp = invoke_gcf(url, payload={})
    return resp

@task(retries=2)
def predictive_random_forest(payload):
    """Perform Random Forest prediction on the stock data"""
    url = "https://us-central1-finnhub-pipeline-ba882.cloudfunctions.net/random_forest_prediction"
    resp = invoke_gcf(url, payload=payload)
    return resp

@task(retries=2)
def predictive_xgboost(payload):
    """Perform XGBoost prediction on the stock data"""
    url = ""
    resp = invoke_gcf(url, payload=payload)
    return resp

@task(retries=2)
def rnn_lstm_prediction():
    """Perform RNN/LSTM prediction on the stock data"""
    url = "https://us-central1-finnhub-pipeline-ba882.cloudfunctions.net/lstm_rnn_predictions"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

# Prefect Flow
@flow(name="finance", log_prints=True)
def prediction_flow():
    """The ETL flow which orchestrates Cloud Functions"""

    result = clean_and_prepare_stock_data()
    print("Stock data cleaning and preparation completed")
    
    rolling_window_result = rolling_window()
    print("Rolling window applied to stock data")
    print(f"{rolling_window_result}")
    
    rf_prediction_result = predictive_random_forest(rolling_window_result)
    print("Random Forest prediction completed")
    print(f"{rf_prediction_result}")

    xgboost_prediction_result = predictive_xgboost(rolling_window_result)
    print("XGBoost prediction completed")
    print(f"{xgboost_prediction_result}")

    rnn_lstm_result = rnn_lstm_prediction()
    print("RNN/LSTM prediction completed")
    print(f"{rnn_lstm_result}")

# the job
if __name__ == "__main__":
    prediction_flow()
