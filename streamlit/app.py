import streamlit as st
from google.cloud import bigquery
import pandas as pd
import matplotlib.pyplot as plt

# Initialize BigQuery client
client = bigquery.Client()

# Function to load data from BigQuery
@st.cache_data(ttl=600)  # Cache the data for 10 minutes
def load_data():
    query = """
    SELECT * FROM `finnhub-pipeline-ba882.financial_data.trades`
    ORDER BY trade_date DESC
    LIMIT 100
    """
    query_job = client.query(query)
    return query_job.to_dataframe()

# Streamlit App UI
st.title("Stock Market Data Visualization")
st.write("This app displays the latest stock market data from BigQuery.")

# Load data from BigQuery
data = load_data()
st.write("Data from BigQuery:", data)

# Plot the data
st.write("Closing Price Over Time")
plt.figure(figsize=(10, 5))
plt.plot(data['trade_date'], data['close'], marker='o')
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("Stock Close Price Over Time")
plt.xticks(rotation=45)
st.pyplot(plt)
