import functions_framework
from google.cloud import secretmanager
from google.cloud import storage
import duckdb
import json
import pandas as pd
import datetime
import uuid
import requests
import os 

# settings
project_id = 'financial-pipeline-group-6'
secret_id = 'mother_db'
version_id = 'latest'
bucket_name = 'financial-pipeline-group-6-bucket'
finnhub_api_key = os.getenv('FINNHUB_API_KEY')

ingest_timestamp = pd.Timestamp.now()

############################################################### helpers

def parse_published(date_str):
    dt_with_tz = datetime.datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
    dt_naive = dt_with_tz.replace(tzinfo=None)
    timestamp = pd.Timestamp(dt_naive)
    return timestamp

def fetch_finnhub_data(symbol, from_timestamp, to_timestamp):
    """Fetch data from the FinnHub API for a given symbol."""
    api_url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&from={from_timestamp}&to={to_timestamp}&token={finnhub_api_key}"
    
    try:
        response = requests.get(api_url)
        data = response.json()
        if data['s'] != 'ok':
            raise ValueError(f"API returned error for symbol {symbol}: {data}")
        return data
    except Exception as e:
        print(f"Error fetching data from FinnHub: {e}")
        raise

############################################################### main task

@functions_framework.http
def task(request):
    # Parse the request data
    request_json = request.get_json(silent=True)
    print(f"request: {json.dumps(request_json)}")

    # Get the job_id and build the GCS base
    job_id = request_json.get('job_id')
    bucket_name = request_json.get('bucket_name')
    gcs_base = f'gs://{bucket_name}/jobs/{job_id}/'

    # FinnHub Data Extraction Parameters
    symbol = request_json.get('symbol', 'AAPL')
    from_timestamp = request_json.get('from_timestamp', 1672531200) #January 1, 2023
    to_timestamp = request_json.get('to_timestamp', 1704067200) #January 1, 2024

    # Fetch data from FinnHub API
    try:
        finnhub_data = fetch_finnhub_data(symbol, from_timestamp, to_timestamp)
    except Exception as e:
        return {"error": str(e)}, 500

    # Convert the fetched data into a DataFrame
    finnhub_df = pd.DataFrame({
        'date': pd.to_datetime(finnhub_data['t'], unit='s'),
        'open': finnhub_data['o'],
        'high': finnhub_data['h'],
        'low': finnhub_data['l'],
        'close': finnhub_data['c'],
        'volume': finnhub_data['v'],
        'job_id': job_id
    })

    finnhub_df['ingest_timestamp'] = ingest_timestamp

    # Write the DataFrame to GCS in Parquet format
    storage_client = storage.Client()

    # Write FinnHub data to GCS
    finnhub_fpath = gcs_base + f"{symbol}_finnhub_data.parquet"
    finnhub_df.to_parquet(finnhub_fpath, engine='pyarrow')

    ########################### return
    return {
        "symbol": symbol,
        "finnhub_data_path": finnhub_fpath,
        "num_entries": len(finnhub_df),
        "job_id": job_id
    }, 200
