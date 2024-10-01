from google.cloud import bigquery
import functions_framework

@functions_framework.http
def main(request):
    bq_client = bigquery.Client()

    dataset_id = 'finhub'
    table_id = 'raw_market_data'
    project_id = bq_client.project

    create_dataset_sql = f"""
    CREATE SCHEMA IF NOT EXISTS `{project_id}.{dataset_id}`
    """

    try:
        bq_client.query(create_dataset_sql).result()
        print(f"Dataset {dataset_id} exists or created successfully.")
    except Exception as e:
        print(f"Error creating dataset {dataset_id}: {e}", 500)

    # EtLT
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.{table_id}` (
        symbol STRING,
        open FLOAT64,
        high FLOAT64,
        low FLOAT64,
        close FLOAT64,
        volume INT64,
        timestamp TIMESTAMP,
        ingest_timestamp TIMESTAMP,
        raw_feed STRING
    )
    """

    # Execute the SQL to create the table
    try:
        bq_client.query(create_table_sql).result()
        print(f"Table {table_id} in dataset {dataset_id} exists or created successfully.")
    except Exception as e:
        print(f"Error creating table {table_id} in dataset {dataset_id}: {e}", 500)
    return {'statusCode':200}
