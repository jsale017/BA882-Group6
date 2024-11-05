import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import matplotlib.pyplot as plt

# Load service account credentials from Streamlit secrets
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["service_account_key"]
)

# Initialize the BigQuery client with credentials
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# Function to load data from BigQuery
@st.cache_data(ttl=600)  # Cache the data for 10 minutes
def load_data():
    query = """
    SELECT * FROM `finnhub-pipeline-ba882.financial_data.trades`
    ORDER BY trade_date DESC
    LIMIT 1200
    """
    query_job = client.query(query)
    return query_job.to_dataframe()

# Streamlit App UI
st.title("Stock Market Data Visualization")
st.write("This app displays the latest stock market data from BigQuery.")

# Load data from BigQuery
data = load_data()

# Show raw data from BigQuery
st.write("### Data from BigQuery", data)

# Visualization: Close Price Over Time
st.write("### Closing Price Over Time")
plt.figure(figsize=(10, 5))
plt.plot(data['trade_date'], data['close'], marker='o', color='blue', label="Close Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("Stock Close Price Over Time")
plt.xticks(rotation=45)
plt.legend()
st.pyplot(plt)

# Visualization: High and Low Prices Over Time
st.write("### Daily High and Low Prices Over Time")
plt.figure(figsize=(10, 5))
plt.plot(data['trade_date'], data['high'], color='green', label="High Price")
plt.plot(data['trade_date'], data['low'], color='red', label="Low Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Daily High and Low Prices Over Time")
plt.xticks(rotation=45)
plt.legend()
st.pyplot(plt)

# Visualization: Volume Over Time
st.write("### Trading Volume Over Time")
plt.figure(figsize=(10, 5))
plt.bar(data['trade_date'], data['volume'], color='purple', alpha=0.7)
plt.xlabel("Date")
plt.ylabel("Volume")
plt.title("Trading Volume Over Time")
plt.xticks(rotation=45)
st.pyplot(plt)

# Data Summary Statistics
st.write("### Data Summary Statistics")
st.write(data.describe())
