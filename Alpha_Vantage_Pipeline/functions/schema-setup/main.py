from google.cloud import bigquery
import functions_framework

@functions_framework.http
def main(request):
    bq_client = bigquery.Client()

    # Define dataset and project ID
    dataset_id = 'financial_data'
    project_id = 'finnhub-pipeline-ba882'
    stock_symbols = ['AAPL', 'NFLX', 'MSFT', 'NVDA', 'AMZN']

    # Creating Dataset in BigQ
    try:
        dataset_ref = bigquery.Dataset(f"{project_id}.{dataset_id}")
        bq_client.create_dataset(dataset_ref, exists_ok=True)
        print(f"Dataset {dataset_id} exists or was created successfully.")
    except Exception as e:
        print(f"Error creating dataset {dataset_id}: {e}")
        return {'statusCode': 500, 'message': f"Error creating dataset: {str(e)}"}

    # Creates Tables for each stock
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
            print(f"Table {symbol.lower()}_prices created or exists successfully.")
        except Exception as e:
            print(f"Error creating table {symbol.lower()}_prices: {e}")
            return {'statusCode': 500, 'message': f"Error creating table: {str(e)}"}

    return {'statusCode': 200, 'message': 'Dataset and tables created successfully'}
