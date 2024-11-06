import argparse
import pandas as pd
from google.cloud import storage
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def download_data(bucket_name, source_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    data = pd.read_csv(blob.download_as_text())
    return data

def train_model(train_data, test_data, target_column, model_type):
    X_train, y_train = train_data.drop(columns=[target_column]), train_data[target_column]
    X_test, y_test = test_data.drop(columns=[target_column]), test_data[target_column]
    
    if model_type == 'xgboost':
        model = XGBRegressor()
    elif model_type == 'random_forest':
        model = RandomForestRegressor()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"{model_type} model for {target_column} MSE: {mse}")
    return model

def main(args):
    train_data = download_data(args.bucket_name, args.train_data)
    test_data = download_data(args.bucket_name, args.test_data)
    
    model = train_model(train_data, test_data, args.target_column, args.model_type)
    joblib.dump(model, f"/tmp/{args.model_type}_{args.target_column}_model.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--target_column", required=True)
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--bucket_name", required=True)
    args = parser.parse_args()
    main(args)
