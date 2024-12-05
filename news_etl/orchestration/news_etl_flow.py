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
    """Transform the news data JSON into tables on GCS"""
    url = "https://transform-news-service-676257416424.us-central1.run.app"
    resp = invoke_gcf(url, payload=payload)
    return resp

@task(retries=2)
def load_news(payload):
    """Load the transformed news data into BigQuery"""
    url = "https://loading-news-676257416424.us-central1.run.app"
    resp = invoke_gcf(url, payload=payload)
    return resp

# Prefect Flow
@flow(name="news_etl_pipeline", log_prints=True)
def news_etl_flow():
    """The ETL flow which orchestrates Cloud Functions for news data"""

    # Step 1: Extract news data
    extract_result = extract_news()
    print("News data extracted successfully")
    print(f"Extract Result: {extract_result}")

    # Step 2: Transform news data
    transform_result = transform_news(extract_result)
    print("News data transformation completed")
    print(f"Transform Result: {transform_result}")

    # Step 3: Load news data
    load_result = load_news(transform_result)
    print("News data loaded into BigQuery")
    print(f"Load Result: {load_result}")

# Entry point
if __name__ == "__main__":
    news_etl_flow()
