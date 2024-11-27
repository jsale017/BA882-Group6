from google.cloud import bigquery

def create_table():
    client = bigquery.Client()

    dataset_id = f"{client.project}.financial_data"
    table_id = f"{dataset_id}.10k_filings"

    # Create dataset if it doesn't exist
    try:
        client.get_dataset(dataset_id)  # Check if dataset exists
        print(f"Dataset {dataset_id} already exists.")
    except Exception:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        client.create_dataset(dataset, exists_ok=True)
        print(f"Dataset {dataset_id} created.")

    # Define table schema
    schema = [
        bigquery.SchemaField("company_name", "STRING"),
        bigquery.SchemaField("cik", "STRING"),
        bigquery.SchemaField("form_type", "STRING"),
        bigquery.SchemaField("filing_date", "STRING"),
        bigquery.SchemaField("document_links", "STRING", mode="REPEATED"),
        bigquery.SchemaField("description", "STRING"),
        bigquery.SchemaField("content", "STRING"),
    ]

    # Create table if it doesn't exist
    try:
        client.get_table(table_id)  # Check if table exists
        print(f"Table {table_id} already exists.")
    except Exception:
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table)
        print(f"Table {table_id} created.")

if __name__ == "__main__":
    create_table()
