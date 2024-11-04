# The ETL job orchestrator

# imports
import requests
import json
from prefect import flow, task

# helper function - generic invoker
def invoke_gcf(url:str, payload:dict):
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


@task(retries=2)
def schema_setup():
    """Setup the stage schema"""
    url = "https://schema-setup-function-676257416424.us-central1.run.app"
    resp = invoke_gcf(url, payload={})
    return resp

@task(retries=2)
def extract():
    """Extract the stock data into JSON on GCS"""
    url = "https://extractv2-676257416424.us-central1.run.app"
    resp = invoke_gcf(url, payload={})
    return resp

@task(retries=2)
def transform(payload):
    """Process the stock data JSON into GCS"""
    url = "https://parsev2-676257416424.us-central1.run.app"
    resp = invoke_gcf(url, payload=payload)
    return resp

@task(retries=2)
def load(payload):
    """Load the tables into the raw schema, ingest new records into stage tables"""
    url = "https://load-data-function-676257416424.us-central1.run.app"
    resp = invoke_gcf(url, payload=payload)
    return resp

# Prefect Flow
@flow(name="finance", log_prints=True)
def etl_flow():
    """The ETL flow which orchestrates Cloud Functions"""

    result = schema_setup()
    print("The schema setup completed")
    
    extract_result = extract()
    print("The Stock Data was extracted onto GCS")
    print(f"{extract_result}")
    
    transform_result = transform(extract_result)
    print("The parsing of the data into tables completed")
    print(f"{transform_result}")

    result = load(transform_result)
    print("The data were loaded into the raw schema and changes added to stage")


# the job
if __name__ == "__main__":
    etl_flow()