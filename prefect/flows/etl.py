import requests
import os
from prefect import flow, task

FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')

if not FINNHUB_API_KEY:
    raise ValueError("Finnhub API Key is not set in the environment variables")

def invoke_gcf(url: str, payload: dict):
    # Simulate GCF invocation for now (you can replace this with real GCF calls later)
    return {"status": "GCF call simulated", "payload": payload}

@task(retries=2, retry_delay_seconds=2)
def schema_setup():
    """Set up the Stage Schema (simulated for now)"""
    url = "YOUR_GCF_URL_FOR_SCHEMA_SETUP"  # Replace with Cloud Function URL if available
    print("Schema setup task triggered")
    resp = invoke_gcf(url, payload={})
    return resp

@task(retries=2, retry_delay_seconds=2)
def extract(symbol: str = "AAPL"):
    """Extract stock data from Finnhub API"""
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        stock_data = response.json()
        print(f"Stock data extracted for {symbol}: {stock_data}")
        return stock_data
    else:
        raise ValueError(f"Failed to fetch stock data for {symbol}: {response.status_code}")

@task(retries=2, retry_delay_seconds=2)
def transform_1(stock_data):
    """Perform first transformation on the stock data"""
    # Example transformation: Add a field to indicate it's transformed
    stock_data['transformed'] = True
    print(f"First transformation: {stock_data}")
    return stock_data

@task(retries=2, retry_delay_seconds=2)
def load(payload):
    """Load the transformed data into your storage (simulated for now)"""
    url = "YOUR_GCF_URL_FOR_LOADING"  # Replace with Cloud Function URL if available
    print("Load task triggered")
    resp = invoke_gcf(url, payload=payload)
    return resp

@task(retries=2, retry_delay_seconds=2)
def transform_2(loaded_data):
    """Perform second transformation on the loaded data"""
    # Example of an additional transformation, let's say we calculate a new field, e.g., price range (high-low)
    loaded_data['price_range'] = loaded_data['h'] - loaded_data['l']
    print(f"Second transformation: {loaded_data}")
    return loaded_data

# Prefect flow to orchestrate the ETLT pipeline
@flow(name='finhub-etlt-flow', log_prints=True)
def etlt_flow(symbol: str = "AAPL"):
    """The ETLT flow to orchestrate Finnhub stock data extraction"""

    # Step 1: Schema setup (if applicable)
    result = schema_setup()  # Simulate schema setup
    print('The Schema Setup is Completed!')

    # Step 2: Extract stock data
    extract_result = extract(symbol)
    print(f"The stock data for {symbol} was extracted!")
    print(f"{extract_result}")

    # Step 3: First transformation
    transform_1_result = transform_1(extract_result)
    print("The first transformation is completed!")
    print(f"{transform_1_result}")

    # Step 4: Load transformed data
    load_result = load(transform_1_result)
    print("The transformed stock data was loaded!")
    print(f"{load_result}")

    # Step 5: Second transformation after loading
    transform_2_result = transform_2(transform_1_result)  # Optionally, load_result can be used
    print("The second transformation is completed!")
    print(f"{transform_2_result}")

if __name__ == "__main__":
    etlt_flow("AAPL")
