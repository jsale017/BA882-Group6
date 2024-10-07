import functions_framework
import os
import requests
from google.cloud import secretmanager
import duckdb
import datetime

# settings
project_id = 'financial-pipeline-group-6'
secret_id = 'mother_db'
version_id = 'latest'
finnhub_api_key = os.getenv('FINNHUB_API_KEY')

# db setup
db = 'financial_data'
schema = "stage"
db_schema = f"{db}.{schema}"

# Rename 'task' to 'finnhub-schema-setup' to match what Cloud Functions is looking for
@functions_framework.http
def finnhub_schema_setup(request):
    try:
        # instantiate the services 
        sm = secretmanager.SecretManagerServiceClient()

        # Build the resource name of the secret version
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

        # Access the secret version
        response = sm.access_secret_version(request={"name": name})
        md_token = response.payload.data.decode("UTF-8")

        # initiate the MotherDuck connection through an access token
        md = duckdb.connect()
        md.sql(f"Set motherduck_token = '{md_token}'") 

        # Create the schema
        create_db_sql = f"CREATE DATABASE IF NOT EXISTS {db};"
        md.sql(create_db_sql)

        # Confirm it exists
        print(md.sql("SHOW DATABASES").show())

        # Create the schema
        md.sql(f"CREATE SCHEMA IF NOT EXISTS {db_schema};") 

        # Create a table for financial data
        financial_tbl_name = f"{db_schema}.financial_data"
        financial_tbl_sql = f"""
        CREATE TABLE IF NOT EXISTS {financial_tbl_name} (
            symbol VARCHAR,
            date TIMESTAMP,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume INT,
            ingest_timestamp TIMESTAMP,
            PRIMARY KEY (symbol, date)
        );
        """
        print(f"{financial_tbl_sql}")
        md.sql(financial_tbl_sql)

        # Extract data from FinnHub
        symbol = "AAPL"
        api_url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&from=1672531200&to=1704067200&token={finnhub_api_key}"
        
        # Fetch stock data
        response = requests.get(api_url)
        data = response.json()

        if data and 's' in data and data['s'] == 'ok':
            for i in range(len(data['t'])):
                date = datetime.datetime.utcfromtimestamp(data['t'][i]).strftime('%Y-%m-%d %H:%M:%S')
                open_price = data['o'][i]
                high_price = data['h'][i]
                low_price = data['l'][i]
                close_price = data['c'][i]
                volume = data['v'][i]

                insert_sql = f"""
                INSERT INTO {financial_tbl_name} (symbol, date, open, high, low, close, volume, ingest_timestamp)
                VALUES ('{symbol}', '{date}', {open_price}, {high_price}, {low_price}, {close_price}, {volume}, CURRENT_TIMESTAMP);
                """
                print(f"{insert_sql}")
                md.sql(insert_sql)

        return "Data processed successfully", 200

    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}", 500
