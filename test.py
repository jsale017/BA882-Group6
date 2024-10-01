import os
import requests

# Fetch the API key
api_key = os.getenv('FINNHUB_API_KEY')

# Check if the API key is set
if api_key:
    print(f"API Key is set: {api_key}")

    # Test Finnhub API call to get stock quote for AAPL
    url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={api_key}"

    response = requests.get(url)
    
    # Check the response status
    if response.status_code == 200:
        print("API request successful!")
        print(response.json())
    else:
        print(f"API request failed with status code {response.status_code}")
else:
    print("API key is not set.")
