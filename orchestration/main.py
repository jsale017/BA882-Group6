# The News ETL job orchestrator

# imports
import requests
from prefect import flow, task

# Helper function - generic invoker for Cloud Functions
def invoke_gcf(url: str, payload: dict):
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

@task(retries=2)
def extract_news():
    """Extract the news data into JSON on GCS"""
    url = "https://extract-news-service-676257416424.us-central1.run.app"
    resp = invoke_gcf(url, payload={})
    return resp

@task(retries=2)
def transform_news(payload):
    """Process the news data JSON into GCS"""
    url = "https://transform-news-service-676257416424.us-central1.run.app"
    resp = invoke_gcf(url, payload=payload)
    return resp

@task(retries=2)
def load_news(payload):
    """Load the transformed news data into BigQuery"""
    url = "https://load-news-service-676257416424.us-central1.run.app"
    resp = invoke_gcf(url, payload=payload)
    return resp

# Prefect Flow
@flow(name="news_etl", log_prints=True)
def etl_news_flow():
    """The ETL flow which orchestrates Cloud Functions for news data"""

    # Step 1: Extract news data
    extract_result = extract_news()
    print("The News Data was extracted onto GCS")
    print(f"Extract Result: {extract_result}")

    # Step 2: Transform news data
    transform_result = transform_news(extract_result)
    print("The parsing of the news data into tables completed")
    print(f"Transform Result: {transform_result}")

    # Step 3: Load news data
    load_result = load_news(transform_result)
    print("The news data was loaded into BigQuery")
    print(f"Load Result: {load_result}")


# Execute the ETL flow
if __name__ == "__main__":
    etl_news_flow()
