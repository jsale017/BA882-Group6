import logging
from google.cloud import bigquery
import functions_framework

logging.basicConfig(level=logging.INFO)

def setup_schema():
    """Set up BigQuery schema: dataset and tables."""
    bq_client = bigquery.Client()
    project_id = 'finnhub-pipeline-ba882'
    dataset_id = 'financial_data'
    stock_symbols = ['AAPL', 'NFLX', 'MSFT', 'NVDA', 'AMZN']

    try:
        # Creating dataset
        dataset_ref = bigquery.Dataset(f"{project_id}.{dataset_id}")
        bq_client.create_dataset(dataset_ref, exists_ok=True)
        logging.info(f"Dataset {dataset_id} created or already exists.")
    except Exception as e:
        logging.error(f"Error creating dataset {dataset_id}: {e}")
        raise

    # Creating tables for each stock symbol
    for symbol in stock_symbols:
        table_id = f"{project_id}.{dataset_id}.{symbol.lower()}_prices"
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS `{table_id}` (
            date STRING,
            open FLOAT64,
            high FLOAT64,
            low FLOAT64,
            close FLOAT64,
            volume INT64
        )
        """
        try:
            bq_client.query(create_table_sql).result()
            logging.info(f"Table {symbol.lower()}_prices created or exists.")
        except Exception as e:
            logging.error(f"Error creating table {symbol.lower()}_prices: {e}")
            raise

@functions_framework.http
def main(request):
    """HTTP Cloud Function entry point."""
    try:
        setup_schema()
        return {"statusCode": 200, "message": "Schema setup complete"}
    except Exception as e:
        logging.error(f"Schema setup failed: {str(e)}")
        return {"statusCode": 500, "message": f"Error: {str(e)}"}
