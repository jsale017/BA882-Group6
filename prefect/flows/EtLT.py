# imports
import requests
import json
from prefect import flow, task

# helper function - generic invoker
def invoke_gcf(url: str, payload: dict):
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

@task(retries=2)
def schema_setup():
    """Setup the stage schema for FinnHub data"""
    url = ""
    resp = invoke_gcf(url, payload={})
    return resp

@task(retries=2)
def extract():
    """Extract the financial data from FinnHub and store it on GCS"""
    url = ""
    payload = {
        "symbol": "AAPL",  # Example stock symbol
        "from_timestamp": 1672531200,  # Example start time (UNIX timestamp)
        "to_timestamp": 1704067200     # Example end time (UNIX timestamp)
    }
    resp = invoke_gcf(url, payload=payload)
    return resp

@task(retries=2)
def transform(payload):
    """Process the financial data into parquet format on GCS"""
    url = ""
    resp = invoke_gcf(url, payload=payload)
    return resp

@task(retries=2)
def load(payload):
    """Load the financial data into the raw schema and ingest new records into stage tables"""
    url = ""
    resp = invoke_gcf(url, payload=payload)
    return resp

# Prefect Flow
@flow(name="finnhub-etl-flow", log_prints=True)
def etl_flow():
    """The ETL flow which orchestrates the financial data pipeline using Google Cloud Functions"""

    # Step 1: Setup schema
    result = schema_setup()
    print("The schema setup completed")

    # Step 2: Extract financial data from FinnHub API and store it in GCS
    extract_result = extract()
    print("The financial data was extracted from FinnHub and stored on GCS")
    print(f"{extract_result}")

    # Step 3: Transform the data (e.g., converting it to parquet format)
    transform_result = transform(extract_result)
    print("The parsing and transformation of the data into tables completed")
    print(f"{transform_result}")

    # Step 4: Load the transformed data into the raw and stage schemas
    load_result = load(transform_result)
    print("The data was loaded into the raw schema and changes added to the stage schema")

# the job
if __name__ == "__main__":
    etl_flow()
