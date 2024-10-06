# imports
import functions_framework
from google.cloud import secretmanager
from google.cloud import storage
import json
import duckdb
import pandas as pd
import os

# setup
project_id = 'financial-pipeline-group-6'
secret_id = 'mother_db'
version_id = 'latest'

# db setup
db = 'finnhub_data'
schema = "raw"
raw_db_schema = f"{db}.{schema}"
stage_db_schema = f"{db}.stage"

############################################################### main task

@functions_framework.http
def task(request):

    # Parse the request data
    request_json = request.get_json(silent=True)
    print(f"request: {json.dumps(request_json)}")

    # instantiate the services 
    sm = secretmanager.SecretManagerServiceClient()
    storage_client = storage.Client()

    # Build the resource name of the secret version
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    # Access the secret version
    response = sm.access_secret_version(request={"name": name})
    md_token = response.payload.data.decode("UTF-8")

    # initiate the MotherDuck connection through an access token
    md = duckdb.connect(f'md:?motherduck_token={md_token}') 

    # drop if exists and create the raw schema
    create_schema = f"DROP SCHEMA IF EXISTS {raw_db_schema} CASCADE; CREATE SCHEMA IF NOT EXISTS {raw_db_schema};"
    md.sql(create_schema)

    print(md.sql("SHOW DATABASES;").show())

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ tbl: financial data

    # read in from gcs
    financial_data_path = request_json.get('financial_data')
    financial_df = pd.read_parquet(financial_data_path)

    # table logic for financial data
    raw_tbl_name = f"{raw_db_schema}.financial_data"
    raw_tbl_sql = f"""
    DROP TABLE IF EXISTS {raw_tbl_name};
    CREATE TABLE {raw_tbl_name} (
        symbol VARCHAR,
        date TIMESTAMP,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume INT,
        job_id VARCHAR,
        ingest_timestamp TIMESTAMP
    );
    """
    print(f"{raw_tbl_sql}")
    md.sql(raw_tbl_sql)

    # ingest into raw schema
    ingest_sql = f"INSERT INTO {raw_tbl_name} SELECT * FROM financial_df"
    print(f"Import statement: {ingest_sql}")
    md.sql(ingest_sql)
    del financial_df

    # upsert-like operation: will only insert new records, not update
    upsert_sql = f"""
    INSERT INTO {stage_db_schema}.financial_data AS stage
    SELECT *
    FROM {raw_tbl_name} AS raw
    ON CONFLICT (symbol, date)
    DO NOTHING;
    """
    print(upsert_sql)
    md.sql(upsert_sql)

    return {}, 200
