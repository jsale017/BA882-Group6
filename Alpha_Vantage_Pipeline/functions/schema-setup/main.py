import logging
from flask import jsonify
from google.cloud import bigquery

# Initializing logger
logging.basicConfig(level=logging.INFO)

def setup_schema():
    """Set up BigQuery schema: dataset, trades fact table, and year_end_reports dimension table."""
    bq_client = bigquery.Client()
    project_id = 'finnhub-pipeline-ba882'
    dataset_id = 'financial_data'

    # Create dataset
    try:
        dataset_ref = bigquery.Dataset(f"{project_id}.{dataset_id}")
        bq_client.create_dataset(dataset_ref, exists_ok=True)
        logging.info(f"Dataset {dataset_id} created or already exists.")
    except Exception as e:
        logging.error(f"Error creating dataset {dataset_id}: {e}")
        raise

    # Create `trades` fact table
    trades_table_id = f"{project_id}.{dataset_id}.trades"
    create_trades_table_sql = f"""
    CREATE TABLE IF NOT EXISTS `{trades_table_id}` (
        trade_id STRING,
        symbol STRING,
        trade_date DATE,
        open FLOAT64,
        high FLOAT64,
        low FLOAT64,
        close FLOAT64,
        volume INT64
    )
    """
    try:
        logging.info(f"Creating or verifying existence of fact table `{trades_table_id}`.")
        bq_client.query(create_trades_table_sql).result()
        logging.info(f"Fact table 'trades' created or already exists.")
    except Exception as e:
        logging.error(f"Error creating fact table 'trades': {e}")
        raise

    # Create `year_end_reports` dimension table
    year_end_reports_table_id = f"{project_id}.{dataset_id}.year_end_reports"
    create_year_end_reports_table_sql = f"""
    CREATE TABLE IF NOT EXISTS `{year_end_reports_table_id}` (
        symbol STRING,
        year INT64,
        total_revenue FLOAT64,
        net_income FLOAT64,
        eps FLOAT64,
        assets FLOAT64,
        liabilities FLOAT64,
        equity FLOAT64
    )
    """
    try:
        logging.info(f"Creating or verifying existence of dimension table `{year_end_reports_table_id}`.")
        bq_client.query(create_year_end_reports_table_sql).result()
        logging.info(f"Dimension table 'year_end_reports' created or already exists.")
    except Exception as e:
        logging.error(f"Error creating dimension table 'year_end_reports': {e}")
        raise

def main(request):
    """Cloud Function endpoint to set up the schema in BigQuery."""
    try:
        logging.info("Starting schema setup process.")
        setup_schema()
        logging.info("Schema setup completed successfully.")
        return jsonify({"statusCode": 200, "message": "Schema setup complete"})
    except Exception as e:
        logging.error(f"Schema setup failed: {str(e)}")
        return jsonify({"statusCode": 500, "message": f"Error: {str(e)}"}), 500

# Local testing setup (optional)
if __name__ == "__main__":
    from flask import Flask, request
    app = Flask(__name__)
    @app.route("/", methods=["GET", "POST"])
    def local_main():
        return main(request)
    app.run(host="0.0.0.0", port=8080)
