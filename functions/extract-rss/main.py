import functions_framework
from google.cloud import secretmanager, storage
import requests
import json
import datetime
import uuid
import os

# Settings
project_id = 'financial-pipeline-group-6'
secret_id = 'mother_db'
version_id = 'latest'
bucket_name = 'financial-pipeline-group-6-bucket'
finnhub_api_key = os.getenv('FINNHUB_API_KEY2')

# Helper Function: Upload to GCS
def upload_to_gcs(bucket_name, job_id, data):
    """Uploads data to a Google Cloud Storage bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_name = f"jobs/{job_id}/finnhub_data.json"
    blob = bucket.blob(blob_name)

    # Upload the data (here it's a serialized string)
    blob.upload_from_string(data)
    print(f"File {blob_name} uploaded to {bucket_name}.")

    return {'bucket_name': bucket_name, 'blob_name': blob_name}

# Main Function
@functions_framework.http
def finnhub_schema_setup(request):
    job_id = datetime.datetime.now().strftime("%Y%m%d%H%M") + "-" + str(uuid.uuid4())

    try:
        # Check if Finnhub API key is set
        if not finnhub_api_key:
            print("Finnhub API key is missing!")
            return {"error": "Finnhub API key is not set"}, 500

        # Access Secret Manager for MotherDuck token
        sm = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        response = sm.access_secret_version(request={"name": name})
        md_token = response.payload.data.decode("UTF-8")

        # Fetch data from Finnhub API
        symbol = "AAPL"
        api_url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&from=1672531200&to=1704067200&token={finnhub_api_key}"
        try:
            api_response = requests.get(api_url)
            api_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Finnhub: {e}")
            return {"error": f"Error fetching data from Finnhub: {e}"}, 500

        data = api_response.json()

        # Check for valid data response
        if data and 's' in data and data['s'] == 'ok':
            print(f"Data fetched for {symbol}: {len(data['t'])} records")
        else:
            print(f"Invalid data received: {data}")
            return {"error": "Error fetching data from FinnHub"}, 500

        # Upload the data to GCS
        data_json = json.dumps(data)
        gcs_path = upload_to_gcs(bucket_name, job_id, data_json)

        return {
            "num_entries": len(data.get('t', [])),
            "job_id": job_id,
            "bucket_name": gcs_path.get('bucket_name'),
            "blob_name": gcs_path.get('blob_name')
        }, 200

    except Exception as e:
        print(f"General Error: {e}")
        return {"error": str(e)}, 500
