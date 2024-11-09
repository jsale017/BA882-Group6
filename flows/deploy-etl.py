from prefect import flow

if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/jsale017/Predictive-Financial-Analytics-APIs.git",
        entrypoint="Alpha_Vantage_Pipeline/flows/etl.py:etl_flow",
    ).deploy(
        name="stock-etl-flow-v2-local-test",
        work_pool_name="Finance Local Test",
        job_variables={
            "env": {"PROJECT_ID": "finnhub-pipeline-ba882"},
            "pip_packages": [
                "pandas", "requests", "google-cloud-storage", 
                "google-cloud-secret-manager", "google-cloud-bigquery"
            ]
        },
        cron="0 21 * * *",
        tags=["prod"],
        description="ETL flow for processing stock market data from Alpha Vantage",
        version="1.0.0",
    )
