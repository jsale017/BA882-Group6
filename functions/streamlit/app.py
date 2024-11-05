import streamlit as st
from google.cloud import bigquery
import pandas as pd

# Initialize BigQuery Client
client = bigquery.Client()

# Streamlit app
st.title("BigQuery Data Visualization")

# Input SQL Query
query = st.text_area("Enter your SQL query here", "SELECT * FROM `finnhub-pipeline-ba882.financial_data.all_stocks_prices`")

if st.button("Run Query"):
    try:
        # Run the query
        query_job = client.query(query)
        df = query_job.to_dataframe() 
        st.write("Query Results:", df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
